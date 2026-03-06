"""
3D Tumor Growth Simulation
Based on: "Fractal dynamics and cancer growth" - Adrià Terradellas Igual (2019)
Implements Section II: Computational Methods

Variables:
  - c(x,t): oxygen consumption field
  - phi(x,t): pro-angiogenic factor field
  - C = c/cmax: normalized oxygen consumption ratio
  - d: cell death probability  = alpha * C
  - b: cell division probability = beta * (1 + gamma - C)
  - R: ratio b/d (used in strange attractor)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random

# ─────────────────────────────────────────────
#  SIMULATION PARAMETERS (fixed in thesis)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  SIMULATION PARAMETERS
# ─────────────────────────────────────────────
L    = 40          # grid side length (nodes)
C_MIN= 100         # minimum oxygen consumption
C_MAX= 500         # hypoxia threshold
GAMMA= 0.1         # condensing factor (+ for condensing, - for non-condensing)
N_A  = 500         # cell count at which angiogenic switch turns on
D_OX = 2.0         # oxygen diffusion coefficient
D_CH = 1000.0      # pro-angiogenic factor diffusion coefficient
DELTA= 1.2         # effectiveness of pro-angiogenic factor on oxygen restoration
N_OX = 100         # oxygen diffusion time steps per simulation step
N_CH = 50          # chemokine diffusion time steps per simulation step
DT   = 0.0001      # diffusion time step
DX   = 1.0         # lattice spacing

# Tunable intrinsic parameters
ALPHA= 0.3         # resistance factor (max death probability)
BETA = 0.7         # growth factor (max division probability)

MAX_SIM_STEPS = 30    # simulation time steps
SEED = 42

# ─────────────────────────────────────────────
#  CELL REPRESENTATION
# ─────────────────────────────────────────────
class Cell:
    """A single tumor cell on the lattice."""
    def __init__(self, x, y, z, condensing: bool):
        self.x = x
        self.y = y
        self.z = z
        self.condensing = condensing          # True = condensing, False = non-condensing
        self.gamma = GAMMA if condensing else -GAMMA
        self.alive = True

    @property
    def pos(self):
        return (self.x, self.y, self.z)

# ─────────────────────────────────────────────
#  NEIGHBOR OFFSETS (1st + 2nd order, total 18)
# ─────────────────────────────────────────────
def get_neighbors_18():
    """Return all 1st and 2nd order neighbor offsets in a cubic lattice (18 total)."""
    offsets = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                n_nonzero = (dx != 0) + (dy != 0) + (dz != 0)
                if n_nonzero == 1 or n_nonzero == 2:   # 1st order (6) + 2nd order (12)
                    offsets.append((dx, dy, dz))
    return offsets

NEIGHBORS_18 = get_neighbors_18()
NEIGHBORS_6  = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

# ─────────────────────────────────────────────
#  DIFFUSION (2D finite differences, applied per z-slice for speed)
# ─────────────────────────────────────────────
def diffusion_step_2d(field_2d: np.ndarray, D: float, dt: float, dx: float) -> np.ndarray:
    """One explicit finite-difference step of the 2D diffusion equation."""
    u = field_2d
    u_new = u.copy()
    # interior points only (Neumann BC: zero-flux at boundary)
    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + D * dt / dx**2 * (
        u[2:, 1:-1] + u[:-2, 1:-1] +
        u[1:-1, 2:] + u[1:-1, :-2] -
        4 * u[1:-1, 1:-1]
    )
    return u_new


def diffuse_3d(field: np.ndarray, D: float, n_steps: int, dt: float = DT, dx: float = DX) -> np.ndarray:
    """Apply 2D diffusion independently on each z-slice for n_steps."""
    for _ in range(n_steps):
        for z in range(field.shape[2]):
            field[:, :, z] = diffusion_step_2d(field[:, :, z], D, dt, dx)
    return field


# ─────────────────────────────────────────────
#  MAIN SIMULATION CLASS
# ─────────────────────────────────────────────
class TumorSimulation:
    def __init__(self, L=L, alpha=ALPHA, beta=BETA, seed=SEED):
        self.L     = L
        self.alpha = alpha
        self.beta  = beta
        self.rng   = np.random.default_rng(seed)
        random.seed(seed)

        # Lattice: None = empty, Cell object = occupied
        self.lattice = np.full((L, L, L), None, dtype=object)

        # Continuous fields (oxygen consumption and pro-angiogenic factor)
        self.c   = np.zeros((L, L, L))   # oxygen consumption
        self.phi = np.zeros((L, L, L))   # pro-angiogenic factor

        # State tracking
        self.cells         = []
        self.angiogenic_on = False
        self.t             = 0

        # History for plotting
        self.history = {
            'population':        [],
            'metastatic_cells':  [],
            'avg_b':             [],
            'avg_d':             [],
            'avg_C':             [],
            'R_ratio':           [],
        }

        # Seed initial tumor cell at center
        cx, cy, cz = L // 2, L // 2, L // 2
        self._place_cell(cx, cy, cz)

        # Initialise oxygen consumption at center
        self.c[cx, cy, cz] = C_MIN

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _place_cell(self, x, y, z):
        """Create and place a new cell at (x,y,z)."""
        condensing = bool(self.rng.integers(0, 2))
        cell = Cell(x, y, z, condensing)
        self.lattice[x, y, z] = cell
        self.cells.append(cell)

    def _remove_cell(self, cell: Cell):
        cell.alive = False
        self.lattice[cell.x, cell.y, cell.z] = None
        self.cells.remove(cell)

    def _in_bounds(self, x, y, z):
        return 0 <= x < self.L and 0 <= y < self.L and 0 <= z < self.L

    # ── Probability equations (Sec. II A) ────────────────────────────────────

    def C_ratio(self, x, y, z) -> float:
        """Normalized oxygen consumption at position (Eq. 1)."""
        return float(np.clip(self.c[x, y, z] / C_MAX, 0.0, 1.0))

    def death_prob(self, cell: Cell) -> float:
        """Cell death probability (Eq. 2): d = alpha * C."""
        C = self.C_ratio(cell.x, cell.y, cell.z)
        return self.alpha * C

    def division_prob(self, cell: Cell) -> float:
        """Cell division probability (Eq. 3): b = beta * (1 + gamma - C)."""
        C = self.C_ratio(cell.x, cell.y, cell.z)
        val = self.beta * (1.0 + cell.gamma - C)
        return float(np.clip(val, 0.0, 1.0))

    # ── Angiogenic switch & diffusion (Sec. II B-D) ──────────────────────────

    def _update_oxygen(self):
        """Diffuse oxygen and subtract pro-angiogenic contribution (Eq. 5)."""
        self.c = diffuse_3d(self.c.copy(), D_OX, N_OX)
        if self.angiogenic_on:
            self.c -= DELTA * self.phi
            self.c = np.clip(self.c, 0.0, None)

    def _update_phi(self):
        """Release and diffuse pro-angiogenic factors from tumor shell (Eq. 4)."""
        N = len(self.cells)
        # Inject phi on outer ~30% shell of occupied region
        cx, cy, cz = self.L // 2, self.L // 2, self.L // 2
        # Estimate tumor radius from cell count
        r_est = max(1, (3 * N / (4 * np.pi)) ** (1/3))
        shell_inner = r_est * 0.7

        for cell in self.cells:
            dist = np.sqrt((cell.x - cx)**2 + (cell.y - cy)**2 + (cell.z - cz)**2)
            if dist >= shell_inner:
                self.phi[cell.x, cell.y, cell.z] += (N / N_A) * 0.5

        self.phi = diffuse_3d(self.phi.copy(), D_CH * DT, N_CH)

    # ── Choose neighbor for daughter cell ───────────────────────────────────

    def _choose_neighbor(self, x, y, z):
        """
        Pre-angiogenic: uniform random from 18 neighbors.
        Post-angiogenic: prefer neighbor with highest free oxygen (lowest c).
        """
        candidates = [
            (x + dx, y + dy, z + dz)
            for dx, dy, dz in NEIGHBORS_18
            if self._in_bounds(x + dx, y + dy, z + dz)
        ]
        if not candidates:
            return None

        if not self.angiogenic_on:
            return random.choice(candidates)
        else:
            # Chemotaxis: pick neighbor with lowest oxygen consumption
            return min(candidates, key=lambda p: self.c[p[0], p[1], p[2]])

    # ── Metastasis process (Sec. II E) ───────────────────────────────────────

    def _attempt_metastasis(self, x, y, z) -> bool:
        """
        Walk outward from (x,y,z) until an empty site is found.
        If the last occupied site has only a single 1st-order neighbour,
        the daughter cell detaches → metastatic event.
        Returns True if metastatic.
        """
        visited = set()
        current = (x, y, z)
        max_walk = 50  # limit walk length

        for _ in range(max_walk):
            visited.add(current)
            cx, cy, cz = current
            # Count 1st-order occupied neighbours of current
            occ_neighbours = [
                (cx + dx, cy + dy, cz + dz)
                for dx, dy, dz in NEIGHBORS_6
                if self._in_bounds(cx + dx, cy + dy, cz + dz)
                and self.lattice[cx + dx, cy + dy, cz + dz] is not None
            ]
            # Find an empty neighbour
            empty_neighbours = [
                (cx + dx, cy + dy, cz + dz)
                for dx, dy, dz in NEIGHBORS_18
                if self._in_bounds(cx + dx, cy + dy, cz + dz)
                and self.lattice[cx + dx, cy + dy, cz + dz] is None
                and (cx + dx, cy + dy, cz + dz) not in visited
            ]
            if empty_neighbours:
                # Place daughter there
                nx, ny, nz = random.choice(empty_neighbours)
                # Metastatic if last occupied site had only 1 first-order neighbour
                if len(occ_neighbours) <= 1:
                    return True   # detached → metastatic
                self._place_cell(nx, ny, nz)
                return False
            # No empty site: move to a random occupied neighbour
            occupied = [
                (cx + dx, cy + dy, cz + dz)
                for dx, dy, dz in NEIGHBORS_18
                if self._in_bounds(cx + dx, cy + dy, cz + dz)
                and self.lattice[cx + dx, cy + dy, cz + dz] is not None
                and (cx + dx, cy + dy, cz + dz) not in visited
            ]
            if not occupied:
                break
            current = random.choice(occupied)
        return False

    # ── One simulation step ──────────────────────────────────────────────────

    def step(self):
        """Advance simulation by one time step."""
        N = len(self.cells)

        # ── Check angiogenic switch
        if not self.angiogenic_on and N >= N_A:
            self.angiogenic_on = True
            print(f"  [t={self.t}] Angiogenic switch ON  (N={N})")

        # ── Update diffusion fields
        self._update_oxygen()
        if self.angiogenic_on:
            self._update_phi()

        # ── Oxygen concentration: add contribution of all cells
        for cell in self.cells:
            self.c[cell.x, cell.y, cell.z] = min(
                self.c[cell.x, cell.y, cell.z] + 1.0,
                C_MAX + 50
            )

        # ── Cell fate decisions (iterate over copy to avoid mutation issues)
        cells_snapshot = list(self.cells)
        metastatic_count = 0
        b_vals, d_vals, C_vals = [], [], []

        for cell in cells_snapshot:
            if not cell.alive:
                continue

            d = self.death_prob(cell)
            b = self.division_prob(cell)
            C = self.C_ratio(cell.x, cell.y, cell.z)
            b_vals.append(b)
            d_vals.append(d)
            C_vals.append(C)

            roll = self.rng.random()

            if roll < d:
                # Cell death
                self._remove_cell(cell)

            elif roll < d + b:
                # Cell division
                nbr = self._choose_neighbor(cell.x, cell.y, cell.z)
                if nbr is None:
                    continue
                nx, ny, nz = nbr
                if self.lattice[nx, ny, nz] is None:
                    # Empty → place daughter directly
                    self._place_cell(nx, ny, nz)
                else:
                    # Occupied → trigger metastasis walk
                    is_meta = self._attempt_metastasis(nx, ny, nz)
                    if is_meta:
                        metastatic_count += 1

        # ── Record history
        avg_b = float(np.mean(b_vals)) if b_vals else 0.0
        avg_d = float(np.mean(d_vals)) if d_vals else 0.0
        avg_C = float(np.mean(C_vals)) if C_vals else 0.0
        R = avg_b / avg_d if avg_d > 1e-9 else float('inf')

        self.history['population'].append(len(self.cells))
        self.history['metastatic_cells'].append(metastatic_count)
        self.history['avg_b'].append(avg_b)
        self.history['avg_d'].append(avg_d)
        self.history['avg_C'].append(avg_C)
        self.history['R_ratio'].append(min(R, 50))  # clip for plotting

        self.t += 1

    def run(self, n_steps: int = MAX_SIM_STEPS, verbose: bool = True):
        for step_i in range(n_steps):
            self.step()
            if verbose and (step_i % 5 == 0 or step_i == n_steps - 1):
                N = len(self.cells)
                meta = self.history['metastatic_cells'][-1]
                b    = self.history['avg_b'][-1]
                d    = self.history['avg_d'][-1]
                print(f"  t={self.t:3d} | N={N:5d} | meta={meta:3d} | "
                      f"<b>={b:.3f} | <d>={d:.3f} | angio={'ON' if self.angiogenic_on else 'off'}")

# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────

def plot_results(sim: TumorSimulation, fig_path: str = None):
    h = sim.history
    t = np.arange(1, len(h['population']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Tumor Simulation  (α={sim.alpha}, β={BETA}, L={sim.L})",
        fontsize=12
    )

    # 1. Population
    ax = axes[0, 0]
    ax.plot(t, h['population'], color='steelblue', lw=2)
    if sim.angiogenic_on:
        ax.axhline(N_A, color='red', ls='--', lw=1, label=f'Angiogenic switch N={N_A}')
        ax.legend(fontsize=8)
    ax.set_title('Tumor Cell Population')
    ax.set_xlabel('Simulation time')
    ax.set_ylabel('N cells')

    # 2. Division & Death probabilities
    ax = axes[0, 1]
    ax.plot(t, h['avg_b'], color='green',  lw=2, label='<b> division')
    ax.plot(t, h['avg_d'], color='crimson', lw=2, label='<d> death')
    ax.set_title('Mean Cell Probabilities')
    ax.set_xlabel('Simulation time')
    ax.set_ylabel('Probability')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # 3. Oxygen ratio C
    ax = axes[0, 2]
    ax.plot(t, h['avg_C'], color='orange', lw=2)
    ax.axhline(1.0, color='red', ls='--', lw=1, label='Hypoxia threshold (C=1)')
    ax.set_title('Mean Oxygen Consumption Ratio C = c/c_max')
    ax.set_xlabel('Simulation time')
    ax.set_ylabel('C')
    ax.legend(fontsize=8)

    # 4. Metastatic cells per step
    ax = axes[1, 0]
    ax.bar(t, h['metastatic_cells'], color='purple', alpha=0.7)
    ax.set_title('Metastatic Events per Step')
    ax.set_xlabel('Simulation time')
    ax.set_ylabel('Detached cells')

    # 5. R ratio (b/d)
    ax = axes[1, 1]
    ax.plot(t, h['R_ratio'], color='teal', lw=2)
    ax.set_title('Ratio R = <b>/<d> (Division/Death)')
    ax.set_xlabel('Simulation time')
    ax.set_ylabel('R')

    # 6. 3D scatter of current cell positions
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    xs = [c.x for c in sim.cells]
    ys = [c.y for c in sim.cells]
    zs = [c.z for c in sim.cells]
    colors = ['royalblue' if c.condensing else 'tomato' for c in sim.cells]
    ax.scatter(xs, ys, zs, c=colors, s=2, alpha=0.5)
    ax.set_title('Cell Positions\n(blue=condensing, red=non-condensing)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        print(f"\nFigure saved → {fig_path}")
    plt.show()


def plot_oxygen_slice(sim: TumorSimulation, fig_path: str = None):
    """Show oxygen and phi fields at mid-z slice."""
    z_mid = sim.L // 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Diffusion fields at z={z_mid}  (t={sim.t})", fontsize=12)

    im1 = axes[0].imshow(sim.c[:, :, z_mid].T, origin='lower',
                          cmap='hot', aspect='equal')
    axes[0].set_title('Oxygen consumption c(x,y)')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(sim.phi[:, :, z_mid].T, origin='lower',
                          cmap='plasma', aspect='equal')
    axes[1].set_title('Pro-angiogenic factor φ(x,y)')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        print(f"Diffusion figure saved → {fig_path}")
    plt.show()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("3D Tumor Growth Simulation")
    print(f"Parameters: α={ALPHA}, β={BETA}, L={L}")
    print(f"Angiogenic switch at N={N_A} cells")
    print("=" * 60)

    sim = TumorSimulation(L=L, alpha=ALPHA, beta=BETA, seed=SEED)
    sim.run(n_steps=MAX_SIM_STEPS, verbose=True)

    print(f"\nFinal population : {len(sim.cells)} cells")
    print(f"Total metastatic events: {sum(sim.history['metastatic_cells'])}")
    print(f"Angiogenic switch triggered: {sim.angiogenic_on}")

    plot_results(sim, fig_path='tumor_results.png')
    plot_oxygen_slice(sim, fig_path='tumor_diffusion.png')

    # ── Optional: compare multiple (alpha, beta) pairs (like Fig. 2 in the thesis)
    print("\n--- Comparing parameter pairs (α, β) ---")
    combos = [(0.3, 0.5), (0.3, 0.7), (0.3, 0.9), (0.7, 0.5), (0.7, 0.9)]
    fig, ax = plt.subplots(figsize=(10, 6))

    for alpha_i, beta_i in combos:
        # Temporarily override global BETA for each run
        sim_i = TumorSimulation(L=L, alpha=alpha_i, beta=beta_i, seed=SEED)
        # Patch BETA inside division_prob via closure using a subclass trick
        _beta = beta_i
        _alpha = alpha_i
        def _div_prob(cell, a=_alpha, b=_beta):
            C = sim_i.C_ratio(cell.x, cell.y, cell.z)
            return float(np.clip(b * (1.0 + cell.gamma - C), 0.0, 1.0))
        def _dth_prob(cell, a=_alpha):
            C = sim_i.C_ratio(cell.x, cell.y, cell.z)
            return a * C
        import types
        sim_i.division_prob = lambda cell, s=sim_i, b=_beta: float(np.clip(b * (1.0 + cell.gamma - s.C_ratio(cell.x, cell.y, cell.z)), 0.0, 1.0))
        sim_i.death_prob    = lambda cell, s=sim_i, a=_alpha: a * s.C_ratio(cell.x, cell.y, cell.z)
        sim_i.alpha = _alpha
        sim_i.beta  = _beta

        sim_i.run(n_steps=MAX_SIM_STEPS, verbose=False)
        pops = sim_i.history['population']
        ax.plot(range(1, len(pops)+1), pops,
                label=f'α={alpha_i}, β={beta_i}', lw=2)
        print(f"  α={alpha_i}, β={beta_i}: final N={len(sim_i.cells)}, "
              f"meta={sum(sim_i.history['metastatic_cells'])}")

    ax.axhline(N_A, color='k', ls='--', lw=1, label=f'Angiogenic switch N={N_A}')
    ax.set_title('Population of simulated tumor evolutions for different (α, β) pairs')
    ax.set_xlabel('Simulation time')
    ax.set_ylabel('Population')
    ax.legend()
    plt.tight_layout()
    fig.savefig('tumor_comparison.png', dpi=120, bbox_inches='tight')
    print("\nComparison figure saved → tumor_comparison.png")
    plt.show()