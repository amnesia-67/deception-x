# vjepa2 LoRA tuning on Real-Life Trials

Pooled OOF AUROC **0.468** (95% CI 0.362-0.640); per-fold AUROC 0.509 +- 0.168; accuracy@0.5 **0.512** (95% CI 0.374-0.702), n=121

| fold | test AUROC | test acc | n | config | inner-val AUROC | final model | attempts |
|---|---|---|---|---|---|---|---|
| 0 | 0.412 | 0.333 | 30 | r8_last4_lr2e-4 | 0.714 | refit | r8_last4_lr2e-4:ok |
| 1 | 0.697 | 0.739 | 23 | r8_last4_lr2e-4 | 0.800 | refit | r8_last4_lr2e-4:ok |
| 2 | 0.280 | 0.522 | 23 | r16_last8_lr3e-4 | 0.847 | refit | r8_last4_lr2e-4:sub-par; r8_last4_lr5e-5:sub-par; r16_last8_lr3e-4:ok |
| 3 | 0.447 | 0.391 | 23 | r16_last8_lr3e-4 | 0.625 | refit | r8_last4_lr2e-4:sub-par; r8_last4_lr5e-5:sub-par; r16_last8_lr3e-4:ok |
| 4 | 0.708 | 0.636 | 22 | r8_last4_lr2e-4 | 0.800 | refit | r8_last4_lr2e-4:ok |
