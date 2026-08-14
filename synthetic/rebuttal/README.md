# Rebuttal experiments

These are the two synthetic benchmark experiments reported in the NeurIPS 2026
rebuttal. Each directory contains the executable configuration and a compact
copy of the final table, summaries, and plots.

## Experiment I: continuous-state OPE benchmarks

Experiment I evaluates 18 configurations: two horizons, three target policies,
and three target embeddings. It uses `n=2,000`, `k=5`, 300 paired repetitions,
known propensities, and independent target-policy trajectories for scoring.
Regression penalties for DR and FH-MIS are selected using logged data only.

```bash
python synthetic/rebuttal/experiment_1/run_benchmark_grid.py \
  --jobs 8 \
  --output-dir synthetic/rebuttal/experiment_1/outputs/main
```

The command writes:

- `experiment_1_table.md`: the six-row table reported in the rebuttal;
- `benchmark_mse_ratios.png`: the 18-cell benchmark heatmap;
- `replicate_estimates.csv`: paired estimates for all methods;
- method, comparator, and reference-value summaries.

The final reported outputs are preserved in
[`experiment_1/reference_results`](experiment_1/reference_results/).
They can be replotted directly with:

```bash
python synthetic/rebuttal/experiment_1/replot_reference_results.py
```

## Experiment II: state dimension and coverage

Experiment II fixes `T=160`, `n=2,000`, and `k=5`, uses ten target embeddings,
and varies `d_x` over 4, 8, 12, and 16. FH-MIS is computed exactly on the
finite augmented state space; Sequential DR uses two-fold cross-fitted linear
continuation-value models.

First compute the exact target values, then run the 100 paired repetitions and
build the figures:

```bash
python synthetic/rebuttal/experiment_2/run_state_dimension_experiment.py \
  --truth-only \
  --output-dir synthetic/rebuttal/experiment_2/outputs/main/truth

python synthetic/rebuttal/experiment_2/run_state_dimension_experiment.py \
  --device auto \
  --output-dir synthetic/rebuttal/experiment_2/outputs/main/replicates

python synthetic/rebuttal/experiment_2/plot_results.py \
  --run-dir synthetic/rebuttal/experiment_2/outputs/main
```

The plotting command writes the dimension table, MSE-ratio plots, RMSE curves,
state-coverage diagnostics, and intervention-survival diagnostics. The final
reported outputs are in [`experiment_2/reference_results`](experiment_2/reference_results/).

For a quick implementation check:

```bash
python synthetic/rebuttal/experiment_2/run_state_dimension_experiment.py \
  --self-test \
  --output-dir /tmp/one_shot_ope_self_test
```
