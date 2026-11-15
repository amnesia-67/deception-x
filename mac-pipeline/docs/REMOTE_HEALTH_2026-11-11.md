# Remote health addendum — reachable Windows endpoint, source still unmatched

Root supplied materially new metadata-only evidence after the source audit. This child made no endpoint calls, transmitted no media, ran no model and changed no production files.

## New evidence

The verified Windows LAN address `192.168.0.224:5002` returned a healthy response in **18.943 ms** during root's bounded ping. The separately supplied `172.23.36.212` is the WSL-internal address and was not reachable from the Mac in the prior check. This resolves the immediate LAN reachability issue for the Windows endpoint.

Reported health fields include:

- Tabular model `v2_tabular_xgboost`,89 features, CUDA device.
- `jepa_loaded:true`.
- `jepa.model:"vjepa2_vitl_lora_5fold"`,5 folds.
- `adapter_shapes:[[8,4],[16,8]]`.
- Reported cross-validation AUROC approximately 0.468.
- Advertised models `["v2_tabular_xgboost","vjepa2_vitl_lora_5fold"]`.

These are **service declarations**, not independently verified checkpoint identities, adapter load completeness, CUDA forward execution or model quality. The 18.943 ms measures health roundtrip only; it does not measure real video encoding or fresh-window throughput. The meaning of `adapter_shapes` needs actual server/export source; do not infer tensor dimensions or LoRA configuration from that field alone.

## What the new evidence changes

The running peer has a different health schema/model label from the committed `src/v2_socket_server.py` inspected at `9c11432c953dcbe653d637dcd2f2814ba7590d57`. Therefore that committed entry point is not a sufficient description of the actual running server. The prior finding “no raw-video endpoint in the inspected source” remains valid **for those files**, but cannot determine whether the different live process implements one.

The user explicitly accepts **V-JEPA2.1 ViT-B temporarily**, until matching ViT-L training is complete. That authorizes a correctly named, versioned ViT-B contract; it does not make ViT-B/768 interchangeable with ViT-L/1024. The reachable service currently labels its encoder **V-JEPA2 ViT-L**, not V-JEPA2.1 ViT-B or V-JEPA2.1 ViT-L. The actual architecture must be resolved from its loader, base checkpoint/config and exported adapter manifest before selecting the Mac client or feature dimensions.

## Remaining handoff and activation requirements

1. Push the exact running server/client source and provide its launch command plus source revision. Review its actual action/framing, JPEG/pixel format, authentication/size/deadline/queue limits, required timestamp fields and response schema.
2. Confirm its real base model family (2 versus 2.1), variant (B versus L), frame count, resolution, token width and temporal pooling. Record full checkpoint/source revisions and file hashes; verify the exact loaded LoRA key sets/configuration for all5 folds. If it returns an ensemble rather than one feature sequence, define that output explicitly rather than treating a score as encoder features.
3. Review matching retrained export and any fusion/projector/normalization state. The public health label alone cannot resolve the previously observed new 1792/1792 fusion versus legacy 1024/768 deployment mismatch.
4. Add a distinct reviewed adapter for the actual raw-video protocol, preserving true identity. If ViT-B is selected, use its real 32-frame/768-native representation and documented 16-to-32 temporal interpolation. Do not relabel or pad those features to satisfy the old ViT-L contract.
5. Run a non-person synthetic pixel fixture through the genuine remote encoder, including repeat/changed-input and timing checks. Then perform the user's authorized live test once local artifacts, remote contract and end-to-end source-window alignment are ready.

The prepared strict remote client remains disabled. Do not weaken it to accept an undocumented response or send guessed frame payloads. No additional peer probing is required from this child audit while the friend publishes the actual protocol.
