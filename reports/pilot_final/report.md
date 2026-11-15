# Final pilot tuning (sunhacks-webcam-pilot-850-v3-2026-11-15)

Chosen on validation only: conditions {'dasheng': 'bce', 'emotion2vec': 'pairwise', 'vjepa2_1_l': 'bce'}; combination ['dasheng']. Cutoff 0.4863 on the Platt-calibrated fused score. 'session (causal)' = each answer vs the median of that person's answers so far in recording order, no verdict for the first 3 (src.pilot_webcam.SessionBaseline).

| set | n | AUROC (95% CI) | acc (global cutoff) | bal acc | lies caught | truths passed | acc (session, causal) | answers with a session verdict |
|---|---|---|---|---|---|---|---|---|
| validation (selection; optimistic) | 120 | 0.724 (0.63–0.81) | 0.67 | 0.67 | 40/60 | 40/60 | 0.64 | 108 |
| test: all 4 familiar people | 80 | 0.802 (0.70–0.89) | 0.74 | 0.74 | 31/40 | 28/40 | 0.72 | 68 |
| test: previous 40 (used before) | 40 | 0.734 (0.57–0.88) | 0.70 | 0.70 | 13/20 | 15/20 | 0.61 | 28 |
| test: new 40 (clean) | 40 | 0.868 (0.74–0.97) | 0.78 | 0.78 | 18/20 | 13/20 | 0.71 | 28 |
| P5: unseen person (50) | 50 | 0.574 (0.42–0.74) | 0.50 | 0.50 | 3/25 | 22/25 | 0.53 | 47 |

| person (test rows; P5 = all 50 unseen) | n | AUROC | called Lie (global) | bal acc (global) | bal acc (session, median of all answers) |
|---|---|---|---|---|---|
| P1 | 20 | 0.780 | 50% | 0.70 | 0.70 |
| P2 | 20 | 0.780 | 45% | 0.75 | 0.70 |
| P3 | 20 | 0.590 | 60% | 0.60 | 0.60 |
| P4 | 20 | 0.990 | 60% | 0.90 | 0.90 |
| P5 | 50 | 0.574 | 12% | 0.50 | 0.56 |

| single branch | AUROC | acc (global) | acc (session, causal) |
|---|---|---|---|
| dasheng (bce) test | 0.802 | 0.74 | 0.72 |
| dasheng (bce) unseen | 0.574 | 0.50 | 0.53 |
| emotion2vec (pairwise) test | 0.652 | 0.57 | 0.59 |
| emotion2vec (pairwise) unseen | 0.522 | 0.50 | 0.45 |
| vjepa2_1_l (bce) test | 0.654 | 0.61 | 0.53 |
| vjepa2_1_l (bce) unseen | 0.651 | 0.50 | 0.64 |

## Every training run (selection on validation within-person AUROC)

Methods chosen per branch on validation: dasheng bce, emotion2vec pairwise, ViT-L bce (all three exported); the fused verdict uses dasheng alone.

| branch | method | attempt | outcome |
|---|---|---|---|
| dasheng | bce | `r8_last4_lr2e-4` | epoch 9: within-person 0.704, pooled 0.724 (this method's model) |
| dasheng | pairwise | `r8_last4_lr2e-4` | epoch 11: within-person 0.682, pooled 0.680 (this method's model) |
| emotion2vec | bce | `r8_last4_lr2e-4` | stopped by the sub-par watchdog: not learning: train AUROC 0.529, best val AUROC 0.544 |
| emotion2vec | bce | `r8_last4_lr5e-5` | stopped by the sub-par watchdog: not learning: train AUROC 0.504, best val AUROC 0.482 |
| emotion2vec | bce | `r16_last8_lr3e-4` | epoch 12: within-person 0.629, pooled 0.636 (this method's model) |
| emotion2vec | pairwise | `r8_last4_lr2e-4` | epoch 9: within-person 0.638, pooled 0.620 (this method's model) |
| vjepa2_1_l | bce | `r8_last4_lr2e-4` | stopped by the sub-par watchdog: not learning: train AUROC 0.559, best val AUROC 0.545 |
| vjepa2_1_l | bce | `r8_last4_lr5e-5` | stopped by the sub-par watchdog: not learning: train AUROC 0.507, best val AUROC 0.507 |
| vjepa2_1_l | bce | `r16_last8_lr3e-4` | epoch 10: within-person 0.620, pooled 0.647 (this method's model) |
| vjepa2_1_l | pairwise | `r8_last4_lr2e-4` | stopped by the operator after epoch 3 for the presentation deadline (all work due 11:15); not a watchdog decision. It was still learning: within-person validation AUROC 0.559 at epoch 3 (vs 0.620 for the chosen ViT-L bce run); up to 9 more epochs (~2.5 h) would have overrun the deadline. No state kept, so this condition has no candidate model |

Test and P5 were scored once, after these choices. Each saved model reloaded to exactly its selected validation scores; the TF32-off serving differences are in each manifest's `reproduction.serving_numerics`.
Dataset: sunhacks-webcam-pilot-850-v3-2026-11-15 (not in git). Code: scripts/pilot_final.py at commit 143e924.
