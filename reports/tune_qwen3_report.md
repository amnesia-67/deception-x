# qwen3 LoRA tuning on Real-Life Trials

Pooled OOF AUROC **0.570** (95% CI 0.464-0.691); per-fold AUROC 0.576 +- 0.165; accuracy@0.5 **0.545** (95% CI 0.459-0.652), n=121

| fold | test AUROC | test acc | n | config | inner-val AUROC | final model | attempts |
|---|---|---|---|---|---|---|---|
| 0 | 0.662 | 0.600 | 30 | r8_last4_lr2e-4 | 0.679 | refit | r8_last4_lr2e-4:ok |
| 1 | 0.424 | 0.391 | 23 | r8_last4_lr2e-4 | 0.733 | refit | r8_last4_lr2e-4:ok |
| 2 | 0.333 | 0.478 | 23 | r16_last8_lr3e-4 | 0.986 | refit | r8_last4_lr2e-4:sub-par; r8_last4_lr5e-5:terminated; r16_last8_lr3e-4:ok |
| 3 | 0.712 | 0.609 | 23 | r8_last4_lr2e-4 | 0.667 | refit | r8_last4_lr2e-4:ok |
| 4 | 0.746 | 0.636 | 22 | r16_last8_lr3e-4 | 0.775 | refit | r8_last4_lr2e-4:sub-par; r8_last4_lr5e-5:terminated; r16_last8_lr3e-4:ok |
