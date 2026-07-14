# Objective Relationship Diagnostics

> Auto-generated from **625** parameter combinations (152 on the Pareto front).

---

## 1  Is Dissipation Dominated by Tumour Size?

Log–log Pearson correlations between Dissipation and size metrics:

| Metric | Full dataset | Pareto-front subset |
|--------|-------------|---------------------|
| log(D) vs log(N) | 0.743 | 0.627 |
| log(D) vs log(R) | 0.831 | 0.810 |

Variance decomposition of log(D) into additive components:

| Component | Full (%) | Pareto front (%) |
|-----------|----------|-----------------|
| Geometry R² | 827.6 | 387.3 |
| Necrosis NCF | 7.3 | 3.8 |
| Metastasis MEI | 1.8 | 1.1 |

**Dissipation IS strongly dominated by tumour size.** The geometry term (R²) accounts for 827.6% of log-space variance across all 625 combinations. NCF and MEI act as secondary multipliers but explain comparatively little of the combo-to-combo variability in D.

---

## 2  Does Dissipation Behave as a Transport-Cost Objective?

After partialling out shared tumour-size variation (log N), the residual Pearson correlation between Dissipation and Fitness is:

- **Full dataset**: partial r = 0.484
- **Pareto-front subset**: partial r = 0.682

The non-trivial residual (|r| = 0.484) indicates that **Dissipation retains independent information about Fitness beyond tumour size**, capturing genuine transport-cost heterogeneity.

---

## 3  How Strongly Do MEI and NCF Contribute?

In log-space the dissipation functional decomposes exactly as:
```
log(D) = 2·log(R) + log(1 + λ·NCF) + log(1 + λ·MEI)
```

NCF and MEI together explain 9.2% of log(D) variance across the full grid and 4.9% on the Pareto front.  Their combined weight is smaller on the Pareto front, consistent with Pareto-optimal solutions displaying more heterogeneous necrosis and metastatic phenotypes.

- **Necrosis (NCF)**: 7.3% (full) → 3.8% (front)
- **Metastasis (MEI)**: 1.8% (full) → 1.1% (front)

---

## 4  Full Dataset vs Pareto-Front Subset

Selecting for non-dominated trade-offs changes the relative importance of each dissipation component:

| Component | Δ (front − full) | Interpretation |
|-----------|-----------------|----------------|
| Geometry R² | -440.2% | Less size-driven on front |
| Necrosis NCF | -3.6% | NCF less variable on front |
| Metastasis MEI | -0.7% | MEI less variable on front |

---

## 5  Output Files

| File | Content |
|------|---------|
| `07_dissipation_vs_N.png` | Dissipation vs Population (log–log, both datasets) |
| `07_dissipation_vs_R.png` | Dissipation vs Radius (log–log, both datasets) |
| `07_dissipation_vs_ncf.png` | Dissipation vs NCF (both datasets) |
| `07_dissipation_vs_mei.png` | Dissipation vs MEI (both datasets) |
| `07_dissipation_vs_fitness.png` | Dissipation vs Fitness (both datasets) |
| `07_partial_fitness.png` | Partial correlation after partialling out log(N) |
| `07_pairplot_correlation.png` | Scatter matrix (lower) + Pearson r, full vs Pareto front (upper) |
| `07_variance_decomposition.png` | Variance contributions bar chart |
| `objective_variance_decomposition.csv` | Decomposition table |