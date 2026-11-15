# Mac pipeline handoff

These documents accompany the proposed V-JEPA2.1 ViT-B service correction against source commit `6e30a90391984e8dec392e41de1f6ddb25b3ed6d`.

- [Application and export instructions](VJEPA21_SERVICE_INSTRUCTIONS.md)
- [Reviewable service patch](VJEPA21_SERVICE.patch)
- [Independent patch review](VJEPA21_PATCH_REVIEW.md)
- [Deployment status and handoff (updated after applying the patch)](NEW_DEPLOYMENT_HANDOFF.md)
- [Mac client contract for the best-output `analyze` endpoint](MAC_CLIENT_CONTRACT.md)

The patch selects existing tuned2.1ViT-B folds, validates complete adapter/head states, and adds `--jepa21-only` packaging without training or dataset reads.20CPU fake-backbone tests passed, and `git apply --check` passed against the reviewed source. Real GPU/checkpoint compatibility, throughput and a complete live multimodal run remain unverified.

Publishing these documents does not apply the patch or change the running service. Review the instructions before applying; use `git apply --check` against the actual current checkout.


**Update (Node 2):** the V-JEPA 2.1 patch has been applied and packaged, real-weight GPU parity verified, and the stacked ensemble (`analyze`) added as the recommended endpoint. See the updated handoff.

- [Response to MODEL_PIPELINE_REPAIR_HANDOFF.md: fully nested evaluation, fixed features, provenance](REPAIR_RESPONSE.md)
- [Response to MODEL_PIPELINE_REPAIR_REVIEW_2DAF067.md: ViT-L status, corrected wording, golden fixture](REVIEW_RESPONSE_2DAF067.md)
