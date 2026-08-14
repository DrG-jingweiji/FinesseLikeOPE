# Synthetic experiments

This directory contains the embedding-dependent continuous-state simulator and
the synthetic experiments used in the paper and rebuttal.

The scripts under `paper/` preserve the estimator implementation used to
produce the submitted figures. In that implementation, the weight for $R_t$
includes the ratio for $A_t$. The response benchmark experiments use the
clarified reward timing: $R_t=r(X_t)$ is observed before $A_t$, so its weight
ends at $A_{t-1}$. The two implementations are kept separate so that the
submitted numerical results remain reproducible while the benchmark results
match the stated timing convention.

## Original paper figures

The archived numerical summaries and the exact figures used in the paper are
under [`paper/reference_results`](paper/reference_results/). To re-render all
four plots without rerunning Monte Carlo simulation:

```bash
python synthetic/paper/replot_paper_figures.py
```

The re-rendered files are written to `synthetic/paper/reproduced_figures/`.
They use the archived numerical values but are not presentation-identical;
labels, typography, and layout may differ.

| Paper file | Experiment script | Main output |
|---|---|---|
| `synthetic_bias_var_vs_n.png` | `synthetic/paper/run_multi_z_n_sweep_paper.py` | Bias versus sample size for five target embeddings |
| `synthetic_homo_mse.png` | `synthetic/paper/run_homo_mse_vs_k.py` | Homogeneous MSE versus truncation window |
| `synthetic_bias_var_vs_k.png` | `synthetic/paper/run_near_scale_sweep_paper.py` | Median bias and variance versus truncation window |
| `evaluation_synthetic.png` | `synthetic/paper/run_one_new_z_n_sweep_large_streaming.py` | Policy-ranking experiment up to one million trajectories |

### Full Monte Carlo commands

Run the commands below from the repository root. They use the seeds and
configurations recorded for the paper.

Main-text sample-size figure:

```bash
python synthetic/paper/run_multi_z_n_sweep_paper.py \
  --z-seeds 20260431,20260432,20260433,20260434,20260435 \
  --k-fixed 5 \
  --n-grid 100,200,300,500,700,1000,1500,2000,3000,5000,7000,10000 \
  --repeats 80 --mc-paths 60000 --seed 20260416 \
  --kernel gaussian --h-mode rate --z-near-scale 1 --z-near-radius 0 \
  --no-show-gap-lines --error-bars sd --run-tag paper_bias_vs_n
```

Homogeneous MSE figure:

```bash
python synthetic/paper/run_homo_mse_vs_k.py \
  --policy target_svm_step_late \
  --target-step-b=-0.8 --target-step-low 0.02 --target-step-high 0.95 \
  --z-star=-0.5,0,0 --horizon 24 \
  --n-grid 250,500,1000,2000,5000 \
  --repeats 120 --mc-paths 60000 --seed 20260501 \
  --project-plot-k-max 14 --run-tag paper_homo
```

Heterogeneous bias-variance figure:

```bash
python synthetic/paper/run_near_scale_sweep_paper.py \
  --scales 0.10,0.20,0.30,0.50 \
  --z-seeds 20260431,20260432,20260433,20260434,20260435,20260436,20260437,20260438,20260439,20260440 \
  --horizon 24 --n-grid 200,300,500,700,1000 --n-fixed 1000 \
  --k-fixed 12 --repeats 80 --mc-paths 60000 --seed 20260416 \
  --kernel gaussian --h-mode rate --z-near-radius 0.8 \
  --run-tag paper_bias_var_vs_k
```

Large-sample policy-ranking figure:

```bash
python synthetic/paper/run_one_new_z_n_sweep_large_streaming.py \
  --z-seed 20260505 --horizon 24 --k-fixed 5 \
  --n-grid 100,200,300,500,700,1000,1500,2000,3000,5000,7000,10000,20000,50000,100000,200000,500000,1000000 \
  --repeats 50 --mc-paths 60000 --seed 20260416 \
  --kernel gaussian --h-mode rate --z-near-scale 0.1 --z-near-radius 0 \
  --chunk-size 100000 --quantile-low 0.1 --quantile-high 0.9 \
  --show-gap-line --run-tag paper_policy_comparison
```

## Rebuttal benchmark experiments

The final benchmark grids are under [`rebuttal/`](rebuttal/):

- Experiment I compares Ours with FH IS, FH PDIS, SN-PDIS, Sequential DR,
  and continuous-state FH-MIS in the continuous-state simulator.
- Experiment II varies binary-state dimension to study the effect of declining
  exact-state coverage on Ours, Sequential DR, and exact finite-horizon MIS.

See [`rebuttal/README.md`](rebuttal/README.md) for the exact commands and
reference results.

## Source layout

- `dataPipeline/vector_provider.py`: regime-switching trajectory generator and
  policy oracles.
- `opePlatform/window_is_estimator.py`: submitted-figure implementation of the
  kernel-localized rolling estimator.
- `paper/`: paper figure experiments and archived summaries.
- `rebuttal/`: benchmark implementations and final response configurations.
