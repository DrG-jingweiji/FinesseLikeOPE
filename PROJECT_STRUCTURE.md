# Project Structure

This workspace follows the same separation used by `OneShotTreatmentOPE`.

## Folders
- `dataPipeline/`: event-level simulators and policy/reward oracles.
- `opePlatform/`: estimator implementation used by experiments.
- `shared/`: shared trajectory contracts/types.
- `outputs/reports/`: aggregate report outputs from k-sweeps, n-sweeps, and policy sweeps.
- `outputs/single_instances/`: small smoke-test or one-off generated outputs.

## Root scripts
- `run_faithful_n_sweep.py`: fixed-k sample-size sweep on the FINESSE-faithful simulator.
- `run_homo_finesse_faithful.py`: bias/variance vs k on the FINESSE-faithful simulator.
- `run_faithful_policy_sweep.py`: compare target-policy variants on the FINESSE-faithful simulator.
- `run_homo_event_sim.py`: lightweight synthetic event simulator sanity check.

## Current main experiment
Use `run_faithful_n_sweep.py` for the current paper-style FINESSE-like n-sweep. The current large value-gap setting is:

```powershell
python .\run_faithful_n_sweep.py `
  --n-grid 10,30,100,300,1000 `
  --k-fixed 16 `
  --horizon 24 `
  --repeats 60 `
  --mc-paths 1000 `
  --behavior-policy behavior_account_very_late_t24 `
  --target-policy target_account_early_t24 `
  --fixed-affinity-type Type2 `
  --fixed-payment-strategy-type Type2 `
  --allow-type-transitions `
  --intervention-strength value_extreme `
  --reward-mode treatment_sensitive `
  --jobs 4
```

Outputs are written under `outputs/reports/` by default.

## Notes
- Generated artifacts should stay under `outputs/reports/` or `outputs/single_instances/`.
- The estimator does not depend on simulator internals; it consumes `TrajectoryBatch`, policy probability oracles, and rewards.
