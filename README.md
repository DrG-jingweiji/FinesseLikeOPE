# FINESSE-Like One-Shot Treatment OPE

This workspace tests the rolling truncated IPW estimator in event-level credit-card simulators. It is organized to mirror `OneShotTreatmentOPE`: data generation is separated from the OPE estimator and experiment scripts.

## Modules

Data pipeline (`dataPipeline/`):

- `finesse_faithful.py`: FINESSE-style account simulator using merchant affinities, payment strategies, transition matrices, transaction approvals, payments, due-date logic, and policy-controlled one-shot credit-line treatment.
- `event_sim.py`: small vectorized event simulator used as a sanity check.

OPE platform (`opePlatform/`):

- `window_is_estimator.py`: rolling truncated inverse-propensity estimator and IS effective sample size calculation.

Shared contracts (`shared/`):

- `contracts.py`: `TrajectoryBatch` container used by simulators and estimators.

## Main Scripts

Run the current fixed-k n-sweep:

```powershell
python .\run_faithful_n_sweep.py
```

Run a bias/variance curve over truncation window `k`:

```powershell
python .\run_homo_finesse_faithful.py
```

Run policy comparisons:

```powershell
python .\run_faithful_policy_sweep.py
```

Run the lightweight event-simulator sanity check:

```powershell
python .\run_homo_event_sim.py
```

## Current Large-Gap FINESSE-Like Setup

The current large value-gap configuration uses account-observable policies only:

- behavior policy: `behavior_account_very_late_t24`
- target policy: `target_account_early_t24`
- intervention strength: `value_extreme`
- reward mode: `treatment_sensitive`
- horizon: `T=24`

This setting gives approximately:

```text
V(pi_b) ~= 0.717
V(pi)   ~= 0.821
V(pi)-V(pi_b) ~= 0.10
```

Run it with:

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

## Outputs

All generated outputs now live under one output root:

- `outputs/reports/`: aggregate figures and CSV summaries.
- `outputs/single_instances/`: smoke tests and one-off instance logs.

Generated artifacts are not intended to be edited by hand.
