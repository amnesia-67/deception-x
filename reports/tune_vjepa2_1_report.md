# vjepa2_1 LoRA tuning on Real-Life Trials

Pooled OOF AUROC **0.528** (95% CI 0.377-0.748); per-fold AUROC 0.561 +- 0.214; accuracy@0.5 **0.521** (95% CI 0.385-0.679), n=121

| fold | test AUROC | test acc | n | config | inner-val AUROC | final model | attempts |
|---|---|---|---|---|---|---|---|
| 0 | 0.435 | 0.400 | 30 | r8_last4_lr2e-4 | 0.893 | refit | r8_last4_lr2e-4:ok |
| 1 | 0.939 | 0.783 | 23 | r8_last4_lr2e-4 | 0.467 | refit | r8_last4_lr2e-4:sub-par; r8_last4_lr5e-5:terminated; r16_last8_lr3e-4:sub-par |
| 2 | 0.386 | 0.435 | 23 | r8_last4_lr2e-4 | 0.889 | refit | r8_last4_lr2e-4:ok |
| 3 | 0.386 | 0.391 | 23 | r8_last4_lr2e-4 | 0.562 | refit | r8_last4_lr2e-4:ok |
| 4 | 0.658 | 0.636 | 22 | r8_last4_lr2e-4 | 1.000 | refit | r8_last4_lr2e-4:ok |
