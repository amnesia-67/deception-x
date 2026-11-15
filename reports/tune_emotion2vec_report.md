# emotion2vec LoRA tuning on Real-Life Trials

Pooled OOF AUROC **0.446** (95% CI 0.342-0.563); per-fold AUROC 0.441 +- 0.101; accuracy@0.5 **0.446** (95% CI 0.356-0.547), n=121

| fold | test AUROC | test acc | n | config | inner-val AUROC | final model | attempts |
|---|---|---|---|---|---|---|---|
| 0 | 0.491 | 0.433 | 30 | r8_last4_lr2e-4 | 0.768 | refit | r8_last4_lr2e-4:ok |
| 1 | 0.424 | 0.435 | 23 | r8_last4_lr2e-4 | 0.711 | refit | r8_last4_lr2e-4:ok |
| 2 | 0.583 | 0.565 | 23 | r16_last8_lr3e-4 | 0.764 | refit | r8_last4_lr2e-4:terminated; r8_last4_lr5e-5:terminated; r16_last8_lr3e-4:ok |
| 3 | 0.273 | 0.435 | 23 | r8_last4_lr2e-4 | 0.708 | refit | r8_last4_lr2e-4:ok |
| 4 | 0.433 | 0.364 | 22 | r8_last4_lr2e-4 | 0.562 | refit | r8_last4_lr2e-4:ok |
