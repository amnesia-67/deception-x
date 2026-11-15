# Does vjepa2_1_l help? Fully nested paired ablation on unseen people

**Decision: do not include video in the verdict: no evidence it helps on unseen people.** Delta AUROC (with - without video) -0.005, 95% identity-bootstrap CI -0.019 to +0.010; video better in 28% of resamples.

| rule | AUROC | 95% CI | lies caught | truths passed | called deceptive |
|---|---|---|---|---|---|
| video alone (mean of fold models, @0.5) | 0.648 | 0.538-0.760 | 27/61 | 41/60 | 38% |
| dasheng + digest (served) | 0.778 | 0.676-0.900 | 53/61 | 37/60 | 63% |
| dasheng + digest + video | 0.773 | 0.664-0.903 | 53/61 | 37/60 | 63% |

| outer fold | without video | with video | video weight | inner video AUROC |
|---|---|---|---|---|
| 0 | 0.875 | 0.875 | 0.000 | 0.663 |
| 1 | 0.917 | 0.917 | 0.052 | 0.472 |
| 2 | 0.598 | 0.598 | 0.239 | 0.705 |
| 3 | 0.833 | 0.833 | 0.000 | 0.633 |
| 4 | 0.925 | 0.925 | 0.000 | 0.556 |
