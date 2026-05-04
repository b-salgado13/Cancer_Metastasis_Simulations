"""
batch_sweep.py — Cancer Metastasis Batch Parameter Sweep + Pareto Analysis
===========================================================================
Runs N_RUNS independent simulations for every (alpha, beta, gamma, n_a)
combination and computes three multi-objective Pareto metrics per run:

    FITNESS      = alive / (O_consumed * (1 + λ·meta))          [maximise]
    MEI          = total_metastatic_events / final_population    [minimise]
    NCF          = necrotic_cells / total_cells                  [minimise]
    DISSIPATION  = R² · (1 + λ_necro·NCF) · (1 + λ_meta·MEI)   [minimise]
                   where R = geometric radius of the tumour sphere at the final step,
                   estimated as R = (3·N / 4π)^(1/3) from the final cell count N —
                   the same formula used in _update_phi() for the angiogenic shell.
                   Inspired by transport optimisation in river basin models: measures
                   the energy the tumour wastes sustaining growth against internal
                   resistance (necrosis) and metastatic load.

Parameters swept
----------------
    alpha  — resistance factor (max death probability)
    beta   — growth factor (max division probability)
    gamma  — condensing factor (+ condensing / 0 neutral / − non-condensing)
    n_a    — angiogenic-switch threshold (cell count at which phi production starts)

Outputs (single-node mode)
--------------------------
    raw_runs.csv        — per-step history, one row per (run, timestep)
    run_summary.csv     — per-run objectives, one row per run
    pareto_summary.csv  — per-(α,β,γ,N_A) means + Pareto-front flag

SLURM array-job mode
--------------------
When SLURM_ARRAY_TASK_ID is set in the environment, the script handles only
the one (α,β,γ,N_A) combination assigned to that job index, using all node
CPUs for its N_RUNS runs. Outputs go to combo-specific files. Afterwards, run:

    python batch_sweep.py --merge

to combine all pair files and compute the Pareto front.

Example SLURM submission script (submit_sweep.sh):
    #!/bin/bash
    #SBATCH --job-name=tumor_sweep
    #SBATCH --array=0-224         # 225 combos (5 α × 5 β × 3 γ × 3 N_A)
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=32    # all node CPUs for intra-combo parallelism
    #SBATCH --time=04:00:00
    #SBATCH --mem=16G
    python batch_sweep.py

Usage
-----
    # Single node (runs everything):
    python batch_sweep.py

    # After SLURM array jobs finish:
    python batch_sweep.py --merge

Place this file in the same directory as:
    Cancer_Metastasis.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import importlib.util
import io
import itertools
import os
import pathlib
import random as _random
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  SWEEP PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
ALPHA_VALUES: list[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
BETA_VALUES:  list[float] = [0.4, 0.5, 0.6, 0.7, 0.8]
GAMMA_VALUES: list[float] = [-0.1, 0.0, 0.1]   # condensing factor
N_A_VALUES:   list[int]   = [200, 500, 1000]    # angiogenic-switch threshold

N_RUNS:    int = 100
BASE_SEED: int = 0       # seed for run r = BASE_SEED + r
N_STEPS:   int = 40
L:         int = 40

# ── Objective parameters ──────────────────────────────────────────────────────
LAMBDA:       float = 0.01   # penalty on metastatic events in FITNESS
LAMBDA_NECRO: float = 1.0    # necrotic-resistance weight in DISSIPATION
LAMBDA_META:  float = 1.0    # metastatic-load weight in DISSIPATION

# ── Safety limits ─────────────────────────────────────────────────────────────
MAX_CELLS:        int   = 54_000   # stop run early if population exceeds this
TIMEOUT_PER_RUN:  float = 900.0   # seconds per run before it is abandoned

# ── Pareto warning threshold ───────────────────────────────────────────────────
TIMEOUT_WARN_FRAC: float = 0.05   # warn if >5% of a pair's runs timed out

# ── Output ────────────────────────────────────────────────────────────────────
RAW_CSV:    str = "raw_runs.csv"
SUMM_CSV:   str = "run_summary.csv"
PARETO_CSV: str = "pareto_summary.csv"

MAX_WORKERS:  int | None = None    # None → all available CPUs
SHUFFLE_SEED: int        = 2025

# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMIC IMPORT
# ─────────────────────────────────────────────────────────────────────────────
def _load_simulation_module():
    here = pathlib.Path(__file__).parent
    for name in ("Cancer_Metastasis.py",):
        candidate = here / name
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("tumor_sim", candidate)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(f"Cancer_Metastasis.py not found in {here}")

# ─────────────────────────────────────────────────────────────────────────────
#  TIMEOUT
# ─────────────────────────────────────────────────────────────────────────────
class _TimeoutError(Exception):
    pass

def _run_with_timeout(fn, timeout: float):
    result, exc_box, done = [None], [None], threading.Event()
    def _t():
        try:    result[0] = fn()
        except Exception as e: exc_box[0] = e
        finally: done.set()
    threading.Thread(target=_t, daemon=True).start()
    if not done.wait(timeout=timeout):
        raise _TimeoutError(f"exceeded {timeout:.0f}s")
    if exc_box[0]: raise exc_box[0]
    return result[0]

# ─────────────────────────────────────────────────────────────────────────────
#  PATCHED run() — population cap + oxygen tracking already in Cancer_Metastasis.py
# ─────────────────────────────────────────────────────────────────────────────
def _patched_run(self, n_steps: int = 40, verbose: bool = False):
    """
    Replaces TumorSimulation.run(). Stops early if population exceeds MAX_CELLS
    and pads remaining steps so every run always contributes exactly n_steps rows.
    total_oxygen_consumed is on the sim object and accumulates correctly up to
    the cap point; padded steps add no further oxygen cost.
    """
    for _ in range(n_steps):
        self.step()
        if len(self.cells) > MAX_CELLS:
            last = {k: self.history[k][-1] for k in self.history}
            for _ in range(n_steps - self.t):
                for k, v in last.items():
                    self.history[k].append(v)
                self.t += 1
            break

# ─────────────────────────────────────────────────────────────────────────────
#  OBJECTIVE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def _compute_objectives(sim, status: str) -> dict:
    """
    Extract the four Pareto objectives from a completed simulation.

    Returns a dict with: final_alive, final_necrotic, final_total,
    total_metastatic, total_oxygen_consumed, fitness, mei, ncf, dissipation.

    For timed-out runs all objectives are NaN so they are excluded from Pareto.
    """
    if status == 'timeout':
        nan = float('nan')
        return dict(final_alive=nan, final_necrotic=nan, final_total=nan,
                    total_metastatic=nan, total_oxygen_consumed=nan,
                    fitness=nan, mei=nan, ncf=nan, dissipation=nan)

    final_alive    = sum(1 for c in sim.cells if not c.necrotic)
    final_necrotic = sum(1 for c in sim.cells if c.necrotic)
    final_total    = len(sim.cells)           # alive + uncleaned necrotic
    total_meta     = int(sum(sim.history['metastatic_cells']))
    O_consumed     = float(sim.total_oxygen_consumed)

    # FITNESS — maximise; guard against division by zero
    if O_consumed > 0 and final_total > 0:
        fitness = final_alive / (O_consumed * (1.0 + LAMBDA * total_meta))
    else:
        fitness = 0.0

    # MEI — minimise
    mei = total_meta / final_total if final_total > 0 else 0.0

    # NCF — minimise
    ncf = final_necrotic / final_total if final_total > 0 else 0.0

    # DISSIPATION — minimise
    # R is the geometric tumour radius estimated from final cell count using the
    # same spherical-volume formula as _update_phi():  R = (3*N / 4*pi)^(1/3)
    # Using the final snapshot (not a time-mean) captures the tumour's actual
    # spatial footprint, which sets the transport length scale for the analogy.
    import math
    tumor_radius = max(1.0, (3.0 * final_total / (4.0 * math.pi)) ** (1.0 / 3.0))
    dissipation = tumor_radius ** 2 * (1.0 + LAMBDA_NECRO * ncf) * (1.0 + LAMBDA_META * mei)

    return dict(
        final_alive=final_alive, final_necrotic=final_necrotic,
        final_total=final_total, total_metastatic=total_meta,
        total_oxygen_consumed=round(O_consumed, 4),
        fitness=round(fitness, 8), mei=round(mei, 8), ncf=round(ncf, 8),
        dissipation=round(dissipation, 8),
    )

# ─────────────────────────────────────────────────────────────────────────────
#  WORKER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def _run_single(args: tuple) -> tuple[list[dict], dict, str]:
    """
    Run one simulation.

    Returns
    -------
    history_rows : list[dict]  — per-step rows for raw_runs.csv
    summary_row  : dict        — single-run objectives for run_summary.csv
    status       : str         — 'ok' | 'capped' | 'timeout'
    """
    alpha, beta, gamma, n_a, run_id, seed, n_steps, lattice_L = args

    mod = _load_simulation_module()
    # Inject combo-specific phenotype parameters into the module's global scope
    # before constructing the simulation so that Cell.__init__ and the angiogenic
    # switch threshold pick them up correctly.
    mod.GAMMA = gamma
    mod.N_A   = n_a
    mod.TumorSimulation.run = _patched_run   # inject population cap

    def _do():
        with contextlib.redirect_stdout(io.StringIO()):
            sim = mod.TumorSimulation(L=lattice_L, alpha=alpha, beta=beta, seed=seed)
            sim.run(n_steps=n_steps, verbose=False)
        return sim

    try:
        sim    = _run_with_timeout(_do, TIMEOUT_PER_RUN)
        pops   = sim.history['population']
        status = 'capped' if (pops and pops[-1] >= MAX_CELLS * 0.9) else 'ok'
    except _TimeoutError:
        # Return zero-padded history rows and NaN objectives
        history_rows = [
            dict(alpha=alpha, beta=beta, gamma=gamma, n_a=n_a,
                 run_id=run_id, seed=seed,
                 sim_time=t+1, population=0, metastatic_cells=0,
                 avg_b=0.0, avg_d=0.0, avg_C=0.0, R_ratio=0.0)
            for t in range(n_steps)
        ]
        objs = _compute_objectives(None, 'timeout')
        summary_row = dict(alpha=alpha, beta=beta, gamma=gamma, n_a=n_a,
                           run_id=run_id, seed=seed,
                           status='timeout', **objs)
        return history_rows, summary_row, 'timeout'

    # ── History rows ─────────────────────────────────────────────────────────
    h = sim.history
    history_rows = [
        dict(alpha=alpha, beta=beta, gamma=gamma, n_a=n_a,
             run_id=run_id, seed=seed,
             sim_time=t+1,
             population       = h['population'][t],
             metastatic_cells = h['metastatic_cells'][t],
             avg_b            = round(h['avg_b'][t],   6),
             avg_d            = round(h['avg_d'][t],   6),
             avg_C            = round(h['avg_C'][t],   6),
             R_ratio          = round(h['R_ratio'][t], 6))
        for t in range(len(h['population']))
    ]

    # ── Summary row ──────────────────────────────────────────────────────────
    objs = _compute_objectives(sim, status)
    summary_row = dict(alpha=alpha, beta=beta, gamma=gamma, n_a=n_a,
                       run_id=run_id, seed=seed,
                       status=status, **objs)

    return history_rows, summary_row, status

# ─────────────────────────────────────────────────────────────────────────────
#  PARETO FRONT
# ─────────────────────────────────────────────────────────────────────────────
def _is_dominated(row: dict, others: list[dict]) -> bool:
    """
    Return True if `row` is dominated by any entry in `others`.

    Dominance: another point dominates `row` if it is at least as good in
    all objectives AND strictly better in at least one:
        - FITNESS:     higher is better
        - MEI:         lower is better
        - NCF:         lower is better
        - DISSIPATION: lower is better
    """
    f0, m0, n0, d0 = (row['mean_fitness'], row['mean_mei'],
                      row['mean_ncf'], row['mean_dissipation'])
    for o in others:
        if o is row:
            continue
        f1, m1, n1, d1 = (o['mean_fitness'], o['mean_mei'],
                           o['mean_ncf'], o['mean_dissipation'])
        # o dominates row: weakly better in all, strictly better in ≥1
        if (f1 >= f0 and m1 <= m0 and n1 <= n0 and d1 <= d0 and
                (f1 > f0 or m1 < m0 or n1 < n0 or d1 < d0)):
            return True
    return False

def compute_pareto(summ_rows: list[dict]) -> list[dict]:
    """
    Aggregate run_summary rows by (α,β,γ,N_A), compute mean objectives, flag Pareto front.

    Returns a list of dicts, one per (α,β,γ,N_A) combo, with columns:
        alpha, beta, gamma, n_a,
        n_ok, n_capped, n_timeout, timeout_frac, timeout_warning,
        mean_fitness, std_fitness,
        mean_mei,     std_mei,
        mean_ncf,     std_ncf,
        mean_dissipation, std_dissipation,
        pareto_front
    """
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in summ_rows:
        groups[(r['alpha'], r['beta'], r['gamma'], r['n_a'])].append(r)

    agg = []
    for (alpha, beta, gamma, n_a), rows in sorted(groups.items()):
        ok_rows = [r for r in rows if r['status'] != 'timeout']
        n_ok      = len(ok_rows)
        n_capped  = sum(1 for r in rows if r['status'] == 'capped')
        n_timeout = sum(1 for r in rows if r['status'] == 'timeout')
        timeout_frac = n_timeout / len(rows) if rows else 0.0
        warn = timeout_frac > TIMEOUT_WARN_FRAC

        def _stat(key):
            vals = [r[key] for r in ok_rows if not (isinstance(r[key], float) and
                    r[key] != r[key])]  # exclude NaN
            if not vals:
                return float('nan'), float('nan')
            return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        mf, sf = _stat('fitness')
        mm, sm = _stat('mei')
        mn, sn = _stat('ncf')
        md, sd = _stat('dissipation')

        if warn:
            print(f"  [WARNING] α={alpha}, β={beta}, γ={gamma}, N_A={n_a}: "
                  f"{n_timeout}/{len(rows)} runs timed out ({100*timeout_frac:.1f}%) "
                  f"— Pareto reliability reduced")

        agg.append(dict(
            alpha=alpha, beta=beta, gamma=gamma, n_a=n_a,
            n_ok=n_ok, n_capped=n_capped, n_timeout=n_timeout,
            timeout_frac=round(timeout_frac, 4),
            timeout_warning=warn,
            mean_fitness    =round(mf, 8), std_fitness    =round(sf, 8),
            mean_mei        =round(mm, 8), std_mei        =round(sm, 8),
            mean_ncf        =round(mn, 8), std_ncf        =round(sn, 8),
            mean_dissipation=round(md, 8), std_dissipation=round(sd, 8),
            pareto_front=False,
        ))

    # Exclude combos with NaN means (all runs timed out) from Pareto computation
    valid = [r for r in agg if r['mean_fitness'] == r['mean_fitness']]
    for row in valid:
        row['pareto_front'] = not _is_dominated(row, valid)

    return agg

# ─────────────────────────────────────────────────────────────────────────────
#  SLURM MERGE
# ─────────────────────────────────────────────────────────────────────────────
def _pair_file_tag(alpha, beta, gamma, n_a, prefix):
    """Filename for a SLURM combo-specific output, e.g. 'raw_a0.3_b0.7_g0.1_na500.csv'."""
    return f"{prefix}_a{alpha}_b{beta}_g{gamma}_na{n_a}.csv"

def merge_slurm_outputs():
    """
    Merge all combo-specific CSV files written by SLURM array jobs into
    raw_runs.csv, run_summary.csv, and compute pareto_summary.csv.
    """
    raw_files  = sorted(glob.glob("raw_a*_b*_g*_na*.csv"))
    summ_files = sorted(glob.glob("summ_a*_b*_g*_na*.csv"))

    if not raw_files:
        print("No combo CSV files found (expected raw_a*_b*_g*_na*.csv). Run the array jobs first.")
        return

    print(f"Merging {len(raw_files)} raw files and {len(summ_files)} summary files …")

    # Merge raw history
    raw_header_written = False
    with open(RAW_CSV, 'w', newline='') as fout:
        for path in raw_files:
            with open(path, newline='') as fin:
                reader = csv.reader(fin)
                header = next(reader)
                if not raw_header_written:
                    csv.writer(fout).writerow(header)
                    raw_header_written = True
                for row in reader:
                    csv.writer(fout).writerow(row)
    print(f"  → {RAW_CSV}")

    # Merge summaries and compute Pareto
    summ_rows = []
    if summ_files:
        header = None
        with open(SUMM_CSV, 'w', newline='') as fout:
            for path in summ_files:
                with open(path, newline='') as fin:
                    reader = csv.DictReader(fin)
                    if header is None:
                        header = reader.fieldnames
                        writer = csv.DictWriter(fout, fieldnames=header)
                        writer.writeheader()
                    for row in reader:
                        # Cast numeric fields
                        for k in ('alpha','beta','gamma','n_a','run_id','seed',
                                  'final_alive','final_necrotic','final_total',
                                  'total_metastatic'):
                            if row.get(k) not in (None,'nan',''):
                                try: row[k] = int(float(row[k]))
                                except: pass
                        for k in ('total_oxygen_consumed','fitness','mei','ncf','dissipation'):
                            if row.get(k) not in (None,''):
                                try: row[k] = float(row[k])
                                except: pass
                        writer.writerow(row)
                        summ_rows.append(row)
        print(f"  → {SUMM_CSV}")

    # Pareto
    if summ_rows:
        # Re-parse numeric fields for Pareto computation
        def _f(r, k):
            try: return float(r[k])
            except: return float('nan')
        parsed = [dict(alpha=_f(r,'alpha'), beta=_f(r,'beta'),
                       gamma=_f(r,'gamma'), n_a=_f(r,'n_a'),
                       status=r['status'],
                       fitness=_f(r,'fitness'), mei=_f(r,'mei'), ncf=_f(r,'ncf'),
                       dissipation=_f(r,'dissipation'))
                  for r in summ_rows]
        pareto = compute_pareto(parsed)
        _write_pareto(pareto)
    print("Merge complete.")

def _write_pareto(pareto_rows: list[dict]):
    fields = ['alpha','beta','gamma','n_a',
              'n_ok','n_capped','n_timeout','timeout_frac',
              'timeout_warning',
              'mean_fitness','std_fitness',
              'mean_mei','std_mei',
              'mean_ncf','std_ncf',
              'mean_dissipation','std_dissipation',
              'pareto_front']
    with open(PARETO_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pareto_rows)
    n_front = sum(1 for r in pareto_rows if r['pareto_front'])
    print(f"  → {PARETO_CSV}  ({n_front} pairs on Pareto front)")

# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE-NODE RUN
# ─────────────────────────────────────────────────────────────────────────────
RAW_FIELDS = ['alpha','beta','gamma','n_a','run_id','seed','sim_time',
              'population','metastatic_cells','avg_b','avg_d','avg_C','R_ratio']

SUMM_FIELDS = ['alpha','beta','gamma','n_a','run_id','seed','status',
               'final_alive','final_necrotic','final_total',
               'total_metastatic','total_oxygen_consumed',
               'fitness','mei','ncf','dissipation']

def run_single_node(tasks: list[tuple], raw_path: str, summ_path: str):
    total      = len(tasks)
    completed  = 0
    capped     = 0
    timeouts   = 0
    errors     = 0
    t_start    = time.perf_counter()
    summ_rows  = []

    with open(raw_path, 'w', newline='') as fraw, \
         open(summ_path, 'w', newline='') as fsum:

        raw_writer  = csv.DictWriter(fraw, fieldnames=RAW_FIELDS)
        summ_writer = csv.DictWriter(fsum, fieldnames=SUMM_FIELDS)
        raw_writer.writeheader()
        summ_writer.writeheader()

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_run_single, task): task for task in tasks}

            for future in as_completed(futures):
                task = futures[future]
                try:
                    hist_rows, summ_row, status = future.result()
                    raw_writer.writerows(hist_rows)
                    summ_writer.writerow(summ_row)
                    fraw.flush(); fsum.flush()
                    summ_rows.append(summ_row)
                    completed += 1
                    if status == 'capped':   capped   += 1
                    elif status == 'timeout': timeouts += 1
                except Exception as exc:
                    errors += 1
                    alpha_t, beta_t, gamma_t, n_a_t, run_id_t = (
                        task[0], task[1], task[2], task[3], task[4])
                    print(f"  [ERROR] α={alpha_t}, β={beta_t}, γ={gamma_t}, "
                          f"N_A={n_a_t}, run={run_id_t}: {exc}")

                if completed % 250 == 0 or completed == total:
                    el   = time.perf_counter() - t_start
                    rate = completed / el if el > 0 else 0
                    eta  = (total - completed) / rate if rate > 0 else float('inf')
                    print(f"  [{completed:5d}/{total}]  elapsed={el:6.1f}s  "
                          f"rate={rate:.1f}/s  ETA≈{eta:5.0f}s  "
                          f"(capped={capped}, timeouts={timeouts}, err={errors})")

    return summ_rows

# ─────────────────────────────────────────────────────────────────────────────
#  SLURM SINGLE-PAIR RUN
# ─────────────────────────────────────────────────────────────────────────────
def run_slurm_pair(combo_idx: int, combos: list[tuple]):
    alpha, beta, gamma, n_a = combos[combo_idx]
    tasks = [
        (alpha, beta, gamma, n_a, run_id, BASE_SEED + run_id, N_STEPS, L)
        for run_id in range(N_RUNS)
    ]
    raw_path  = _pair_file_tag(alpha, beta, gamma, n_a, 'raw')
    summ_path = _pair_file_tag(alpha, beta, gamma, n_a, 'summ')

    print(f"SLURM job {combo_idx}: α={alpha}, β={beta}, γ={gamma}, N_A={n_a} — {N_RUNS} runs")
    summ_rows = run_single_node(tasks, raw_path, summ_path)

    # Per-combo timeout warning
    n_to = sum(1 for r in summ_rows if r['status'] == 'timeout')
    frac = n_to / len(summ_rows) if summ_rows else 0
    if frac > TIMEOUT_WARN_FRAC:
        print(f"  [WARNING] {n_to}/{len(summ_rows)} runs timed out ({100*frac:.1f}%) "
              f"for α={alpha}, β={beta}, γ={gamma}, N_A={n_a}")
    print(f"  Wrote {raw_path}, {summ_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge', action='store_true',
                        help='Merge SLURM pair outputs and compute Pareto front')
    args = parser.parse_args()

    if args.merge:
        merge_slurm_outputs()
        sys.exit(0)

    # ── Detect SLURM array mode ───────────────────────────────────────────────
    combos = list(itertools.product(ALPHA_VALUES, BETA_VALUES, GAMMA_VALUES, N_A_VALUES))
    slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')

    if slurm_task_id is not None:
        # ── SLURM mode: handle one (α,β,γ,N_A) combo ─────────────────────────
        combo_idx = int(slurm_task_id)
        if combo_idx >= len(combos):
            print(f"SLURM_ARRAY_TASK_ID={combo_idx} out of range (max {len(combos)-1})")
            sys.exit(1)
        _load_simulation_module()   # fail fast
        run_slurm_pair(combo_idx, combos)

    else:
        # ── Single-node mode: run everything ─────────────────────────────────
        _load_simulation_module()   # fail fast

        tasks = [
            (alpha, beta, gamma, n_a, run_id, BASE_SEED + run_id, N_STEPS, L)
            for (alpha, beta, gamma, n_a) in combos
            for run_id in range(N_RUNS)
        ]
        rng_s = _random.Random(SHUFFLE_SEED)
        rng_s.shuffle(tasks)

        print("=" * 66)
        print("Cancer Metastasis — Batch Parameter Sweep + Pareto")
        print("=" * 66)
        print(f"  α values      : {ALPHA_VALUES}")
        print(f"  β values      : {BETA_VALUES}")
        print(f"  γ values      : {GAMMA_VALUES}")
        print(f"  N_A values    : {N_A_VALUES}")
        print(f"  Combos        : {len(combos)}  "
              f"({len(ALPHA_VALUES)} α × {len(BETA_VALUES)} β × "
              f"{len(GAMMA_VALUES)} γ × {len(N_A_VALUES)} N_A)")
        print(f"  Runs / combo  : {N_RUNS}")
        print(f"  Steps / run   : {N_STEPS}")
        print(f"  Seeds         : {BASE_SEED} … {BASE_SEED + N_RUNS - 1}")
        print(f"  Total sims    : {len(tasks)}")
        print(f"  Workers       : {MAX_WORKERS or os.cpu_count()} processes")
        print(f"  Pop. cap      : {MAX_CELLS:,} cells")
        print(f"  Run timeout   : {TIMEOUT_PER_RUN:.0f}s")
        print(f"  Lambda (fit.) : {LAMBDA}")
        print(f"  λ_necro (dis.): {LAMBDA_NECRO}")
        print(f"  λ_meta  (dis.): {LAMBDA_META}")
        print(f"  Output        : {RAW_CSV}, {SUMM_CSV}, {PARETO_CSV}")
        print("-" * 66)

        t0 = time.perf_counter()
        summ_rows = run_single_node(tasks, RAW_CSV, SUMM_CSV)

        # ── Pareto front ─────────────────────────────────────────────────────
        print("\nComputing Pareto front …")
        pareto = compute_pareto(summ_rows)
        _write_pareto(pareto)

        elapsed = time.perf_counter() - t0
        n_front = sum(1 for r in pareto if r['pareto_front'])
        print("-" * 66)
        print(f"Finished.  Total time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
        print(f"  {RAW_CSV}    — per-step history")
        print(f"  {SUMM_CSV}   — per-run objectives")
        print(f"  {PARETO_CSV} — {n_front}/{len(combos)} combos on Pareto front")