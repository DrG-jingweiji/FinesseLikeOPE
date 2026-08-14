# FINESSE-inspired account simulator

This directory contains a lightweight account-level simulator for one-shot
credit-line interventions. It follows FINESSE's merchant-affinity, payment,
transition, transaction-approval, due-date, and missed-payment mechanics while
exposing known behavior and target policy probabilities for OPE.

The wrapper is vectorized and does not execute FINESSE's Mesa model. Rewards
are observed before the action at the same time index, so the weight for
$R_t$ contains action ratios only through $A_{t-1}$.

## FINESSE dependency

Clone the public FINESSE repository at the revision used by these experiments:

```bash
git clone https://github.com/CapitalOne-Research/FINESSE.git third_party/FINESSE
git -C third_party/FINESSE checkout 1a48b9e6442af577bcc49b99687b5b4a3f38a57b
```

The scripts use `third_party/FINESSE` by default. A different location can be
supplied with `--finesse-root`.

## Main experiment

The large-gap configuration uses account-observable policies, horizon `T=24`,
behavior policy `behavior_account_very_late_t24`, target policy
`target_account_early_t24`, and the treatment-sensitive reward.

```bash
python simulator/run_faithful_n_sweep.py \
  --n-grid 10,30,100,300,1000 \
  --k-fixed 16 --horizon 24 --repeats 60 --mc-paths 1000 \
  --behavior-policy behavior_account_very_late_t24 \
  --target-policy target_account_early_t24 \
  --fixed-affinity-type Type2 --fixed-payment-strategy-type Type2 \
  --allow-type-transitions \
  --intervention-strength value_extreme \
  --reward-mode treatment_sensitive \
  --jobs 4
```

Additional scripts:

| Script | Purpose |
|---|---|
| `run_homo_finesse_faithful.py` | Bias and variance versus truncation window |
| `run_faithful_policy_sweep.py` | Comparison across target policies |
| `run_homo_event_sim.py` | Lightweight event-simulator sanity check |

Outputs are written under `simulator/outputs/reports/` by default.

## Source layout

- `dataPipeline/finesse_faithful.py`: FINESSE-inspired account mechanics and
  policy oracles.
- `dataPipeline/event_sim.py`: small vectorized event simulator.
- `opePlatform/window_is_estimator.py`: rolling truncated estimator.
- `shared/contracts.py`: trajectory container shared by simulators and
  estimators.
