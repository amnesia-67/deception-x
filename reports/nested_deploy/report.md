# Served `analyze` policy - fully nested evaluation on unseen people

fully nested: outer 5-fold identity-grouped CV; inside each outer-train set an inner 5-fold identity-grouped CV re-tunes dasheng LoRA, produces digest OOF, fits the stacker and picks the threshold; outer test scored with the served aggregation.

Residual limitation: Which two branches are served was chosen earlier from the 7-branch stacker on all clips; that choice is not re-made inside the outer folds.

**AUROC 0.778** (95% identity-bootstrap CI 0.676-0.900), n = 121 clips / 57 people. Class 1 = deceptive.

| decision rule | lies caught | truths passed | balanced acc | accuracy | called deceptive | Brier | ECE |
|---|---|---|---|---|---|---|---|
| fixed 0.5 | 46/61 (75%) | 41/60 (68%) | 0.719 | 0.719 | 54% | 0.195 | 0.094 |
| prior-matched threshold RE-FITTED inside each outer fold (mean 0.460) | 53/61 (87%) | 37/60 (62%) | 0.743 | 0.744 | 63% | 0.195 | 0.094 |

The re-fitted row evaluates the threshold-fitting procedure; it is not a measurement at the single cutoff exported for serving (see models/ensemble_deploy.json cv_metrics).

Re-fitted-threshold 95% CIs: lie sensitivity 0.75-0.95, truth specificity 0.45-0.80, balanced accuracy 0.64-0.85. Baselines: always-truthful accuracy 0.496, balanced accuracy 0.500; class-prior Brier 0.250.

| branch as served | AUROC | 95% CI | lies caught @0.5 | truths passed @0.5 |
|---|---|---|---|---|
| tuned_dasheng_mean_of_fold_models | 0.759 | 0.612-0.913 | 64% | 73% |
| xgb_digest_full_refit | 0.763 | 0.681-0.874 | 56% | 77% |

| outer fold | n test | AUROC | threshold | weights (dasheng, digest) | inner dasheng AUROC | inner digest AUROC |
|---|---|---|---|---|---|---|
| 0 | 30 | 0.875 | 0.449 | 0.87, 0.13 | 0.803 | 0.739 |
| 1 | 23 | 0.917 | 0.460 | 0.70, 0.30 | 0.753 | 0.635 |
| 2 | 23 | 0.598 | 0.462 | 0.47, 0.53 | 0.746 | 0.763 |
| 3 | 23 | 0.833 | 0.448 | 0.64, 0.36 | 0.770 | 0.728 |
| 4 | 22 | 0.925 | 0.485 | 0.23, 0.77 | 0.619 | 0.734 |

## Duration shift: dasheng on the central 7 s vs the whole answer

inherited whole-clip labels; central window only; not a 7 s target-domain evaluation; 117 of 121 clips are longer than 7 s.

| audio given to the model | AUROC | lies caught @0.5 | truths passed @0.5 | called deceptive | mean prob |
|---|---|---|---|---|---|
| whole answer (<= 60 s, as trained) | 0.764 | 56% | 83% | 36% | 0.433 |
| central 7 s (live length) | 0.736 | 48% | 85% | 31% | 0.402 |

These are source-domain (courtroom) estimates. They are not webcam accuracy; that needs a frozen, consented, labelled 7-second target-domain test set with unseen people.
