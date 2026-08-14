# Experiment II

This experiment varies the binary-state dimension while holding the other
settings fixed. The full configuration, including seeds and the ten target
embeddings, is in `config.json`.

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

`run_state_dimension_experiment.py` can also split repetitions across devices
with `--repeat-start` and `--repeat-count`. Store each split in a directory
named `shard_<index>` under the common run directory; `plot_results.py` merges
all such files.

The reported table and compact result summaries are under
[`reference_results/`](reference_results/).
