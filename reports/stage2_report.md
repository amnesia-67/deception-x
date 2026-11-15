# Stage 2 - real-data results (5-fold identity-grouped CV, out-of-fold)

Digest feature version `v3-fixed-lag-motion`. Class 1 = deceptive. Rates at threshold 0.5.

| branch | AUROC | 95% CI | accuracy | 95% CI | lies caught (TP/61) | truths passed (TN/60) | called deceptive | Brier |
|---|---|---|---|---|---|---|---|---|
| tuned_vjepa2 | 0.468 | 0.362-0.640 | 0.512 | 0.374-0.702 | 11 (18%) | 51 (85%) | 17% | 0.273 |
| tuned_vjepa2_1 | 0.528 | 0.377-0.748 | 0.521 | 0.385-0.679 | 27 (44%) | 36 (60%) | 42% | 0.250 |
| tuned_dasheng | 0.764 | 0.633-0.924 | 0.694 | 0.579-0.864 | 34 (56%) | 50 (83%) | 36% | 0.200 |
| tuned_emotion2vec | 0.446 | 0.342-0.563 | 0.446 | 0.356-0.547 | 27 (44%) | 27 (45%) | 50% | 0.266 |
| tuned_qwen3 | 0.570 | 0.464-0.691 | 0.545 | 0.459-0.652 | 30 (49%) | 36 (60%) | 45% | 0.256 |
| xgb_digest | 0.763 | 0.681-0.874 | 0.661 | 0.556-0.791 | 34 (56%) | 46 (77%) | 40% | 0.204 |
| fusion_redadare | 0.695 | 0.597-0.808 | 0.620 | 0.475-0.780 | 25 (41%) | 50 (83%) | 29% | 0.282 |
| ENSEMBLE stacked, 7 branches (meta-level nesting only) | 0.724 | 0.605-0.927 | 0.636 | 0.491-0.831 | 32 (52%) | 45 (75%) | 39% | 0.208 |
| ENSEMBLE mean | 0.745 | 0.647-0.877 | 0.628 | 0.478-0.808 | 24 (39%) | 52 (87%) | 26% | 0.218 |

Always-truthful baseline: accuracy 0.496, balanced accuracy 0.500.
The served policy's fully nested result is in runs/nested_deploy/report.md.
