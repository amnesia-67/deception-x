# dasheng LoRA tuning on Real-Life Trials

Pooled OOF AUROC **0.764** (95% CI 0.633-0.924); per-fold AUROC 0.752 +- 0.195; accuracy@0.5 **0.694** (95% CI 0.579-0.864), n=121

| fold | test AUROC | test acc | n | config | inner-val AUROC | final model | attempts |
|---|---|---|---|---|---|---|---|
| 0 | 0.639 | 0.467 | 30 | r8_last4_lr2e-4 | 0.920 | refit | r8_last4_lr2e-4:ok |
| 1 | 0.879 | 0.739 | 23 | r8_last4_lr2e-4 | 0.778 | refit | r8_last4_lr2e-4:ok |
| 2 | 0.439 | 0.609 | 23 | r8_last4_lr2e-4 | 0.903 | refit | r8_last4_lr2e-4:ok |
| 3 | 0.803 | 0.739 | 23 | r8_last4_lr2e-4 | 0.708 | refit | r8_last4_lr2e-4:ok |
| 4 | 1.000 | 1.000 | 22 | r8_last4_lr2e-4 | 1.000 | refit | r8_last4_lr2e-4:ok |
