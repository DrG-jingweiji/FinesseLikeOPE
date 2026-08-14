# Experiment I Rebuttal Table

These are the values reported for Experiment I. Each entry is the geometric
mean across the three target embeddings of
`MSE(benchmark) / MSE(Ours)`. Values above one favor Ours.

| T | Target policy | FH IS | FH PDIS | SN-PDIS | Sequential DR | FH-MIS |
|---:|---|---:|---:|---:|---:|---:|
| 24 | Early logistic | 2.54 | 2.38 | 0.80 | 0.82 | 0.70 |
| 24 | Late logistic | 16.39 | 7.88 | 1.27 | 2.96 | 2.20 |
| 24 | Late step | 52.14 | 21.89 | 1.28 | 5.28 | 1.22 |
| 48 | Early logistic | 5.05 | 4.85 | 0.76 | 0.77 | 0.60 |
| 48 | Late logistic | 44.03 | 32.03 | 1.71 | 8.73 | 3.06 |
| 48 | Late step | 176.55 | 124.22 | 1.87 | 11.95 | 1.66 |

Here FH-MIS is the experiment's two-fold cross-fitted continuous-state,
finite-horizon marginal-ratio regression adaptation.
