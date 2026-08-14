# Off-Policy Evaluation of One-Shot Interventions

Reproduction code for **Off-Policy Evaluation of One-Shot Interventions with
Endogenous Regime Switching**. The repository has two independent parts.

Create a Python environment and install the common dependencies before running
the examples:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 1. Synthetic experiments

[`synthetic/`](synthetic/) contains the continuous-state simulator, the
kernel-localized truncated estimator, all four synthetic figures in the
original paper, and the two benchmark experiments reported in the rebuttal.

The exact four paper figures are archived in the repository. They can also be
re-rendered immediately from the archived numerical summaries. The rerenders
use the same numerical values but are not presentation-identical; labels,
typography, and layout may differ:

```bash
python synthetic/paper/replot_paper_figures.py
```

Full Monte Carlo commands and a figure-to-command table are in
[`synthetic/README.md`](synthetic/README.md). The rebuttal experiments are
documented separately in [`synthetic/rebuttal/README.md`](synthetic/rebuttal/README.md).

## 2. Agent-based simulator

[`simulator/`](simulator/) contains the FINESSE-inspired account simulator and
sample-size, truncation-window, and policy sweeps. It uses the public FINESSE
repository for merchant affinities, payment strategies, and merchant metadata.

Setup and run commands are in [`simulator/README.md`](simulator/README.md).
