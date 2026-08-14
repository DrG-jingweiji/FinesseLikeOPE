# Experiment I

`run_benchmark_grid.py` generates the continuous-state benchmark grid and all
reported summaries in one pass. Its defaults are the reported configuration:
`n=2,000`, 300 paired repetitions, 200,000 target-policy trajectories per
reference value, bandwidth `n^{-1/5}`, truncation window `k=5`, and seed
`20270113`.

```bash
python synthetic/rebuttal/experiment_1/run_benchmark_grid.py \
  --jobs 8 \
  --output-dir synthetic/rebuttal/experiment_1/outputs/main
```

Files:

- `benchmark_estimators.py`: propensity, PDIS, DR, and scoring utilities.
- `continuous_mis.py`: two-fold cross-fitted continuous finite-horizon MIS.
- `run_benchmark_grid.py`: simulation, aggregation, table, and plots.
- `replot_reference_results.py`: recreates the reported table and heatmap from
  the compact reference summary.
- `reference_results/`: the table and plots reported in the rebuttal.

To recreate the reported table and heatmap without rerunning the simulation:

```bash
python synthetic/rebuttal/experiment_1/replot_reference_results.py
```
