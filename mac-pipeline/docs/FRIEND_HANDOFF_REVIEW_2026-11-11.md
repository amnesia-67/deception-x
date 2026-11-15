# Current audit of the teammate handoff

## Current status

The source-code and file-transfer gaps are resolved. The repository is pinned to commit `6a16734c1af7d4a5ff24400b06849241b285d5fb`; the original model classes and forward functions have been reviewed. Both `branch2_xgboost.json` and `ensemble_platt_meta.json` have been retrieved from that repository and are available under `/tmp/sunhacks-friend-source-20261111/models`. They are no longer missing.

The Mac now has a strict compatibility path for the original fusion head **and** its saved audio adapter. Root reports 20 strict adapter tests passed. The live-demo input contract uses real V-JEPA temporal tubelet features, native Dasheng frame features followed by the saved adapter and interpolation, and the first 128 Qwen embedding dimensions with L2 normalization. This is an explicitly quality-unverified integration contract: the provided training route did not extract its training examples from those real encoders.

Branch2 and the supplied ensemble remain intentionally inactive despite their files being available. The Mac does not implement the matching250 real features, and the committed calibration procedure contains target leakage. A separate DeepSeek Flash manager connectivity test completed in 1.59 seconds without sending participant data. That API connectivity result does not validate the fusion head or deception classification.

## What the uploaded files establish

`branch1_fusion.pt` is 3,254,993 bytes, SHA256 `cdeeb2f19c51aa9c614665e9c36554e9b1efe03fca367842ee94beec95820e78`. Independent inspection used PyTorch `weights_only=True`, CPU mapping and memory mapping. The file contains two state dictionaries and 51 finite BF16 tensors: 1,026,569 fusion-network parameters and 592,128 dense audio-adapter parameters. There are no LoRA A/B keys, foundation-model weights, training metrics, base-model identifiers, preprocessing metadata or calibration records in that checkpoint.

Its computed Git blob hash is `241946bce2164589c362505a32b57f3e838c66dc`, matching the pinned repository's `models/branch1_fusion.pt`. This establishes that the uploaded bytes are the committed artifact. It does not certify the accuracy of the associated training report.

The uploaded Dasheng `model.safetensors` is 341,807,392 bytes, SHA256 `adaa439ebec13933501242364a29b7912c2695d0354061b278c873438b2736c3`. It is byte-identical to this Mac's official `mispeech/dasheng-base` snapshot at revision `e0b0a27818689e59010bdd7739475e33090dff11`. Thus this file is unchanged base weights. A separate dense adapter can still have been optimized; unchanged base weights alone do not imply that nothing was trained.

## The documented training path is synthetic

The committed [trainer](https://github.com/amnesia-67/sunhacks-2026/blob/6a16734c1af7d4a5ff24400b06849241b285d5fb/src/trainer.py#L242) constructs 216 synthetic examples with 18 numeric speaker IDs and uses a random 75/25 split for Branch1. The [generator](https://github.com/amnesia-67/sunhacks-2026/blob/6a16734c1af7d4a5ff24400b06849241b285d5fb/src/data_generator.py#L58) creates Gaussian-random visual features, text vectors, mel arrays and tabular vectors, then inserts class-dependent offsets or fixed values using the generated label. Those arrays are not recorded clips, V-JEPA outputs or Qwen transcript embeddings. The assigned IDs do not establish generalization to real speakers; Branch1's random split is not speaker-disjoint LOSO.

The [optimizer list](https://github.com/amnesia-67/sunhacks-2026/blob/6a16734c1af7d4a5ff24400b06849241b285d5fb/src/trainer.py#L282) trains only the dense audio adapter and fusion network. Dasheng's backbone is frozen, and no V-JEPA, Qwen or emotion2vec backbone is trained in this route. There is no LoRA training here. The evidence supports a small head/adapter optimized on a synthetic separability exercise, not a claim that all foundation models were tuned on real task data. If different real-data training occurred elsewhere, its source, preprocessing, artifacts and held-out results have not been supplied in this reviewed route.

The [adversarial evaluation](https://github.com/amnesia-67/sunhacks-2026/blob/6a16734c1af7d4a5ff24400b06849241b285d5fb/src/adversarial_eval.py) also constructs seeded random arrays with manually inserted markers, scripted transcripts and expected labels. Four such scenarios do not establish blind human-subject evaluation or countermeasure robustness. This audit did not run training, generate a dataset or download a dataset.

## Reported calibration and perfect scores are not independent validation

The [trainer's calibration path](https://github.com/amnesia-67/sunhacks-2026/blob/6a16734c1af7d4a5ff24400b06849241b285d5fb/src/trainer.py#L354) generates the simulated cloud branch's probabilities directly from the ground-truth label: positives above 0.65 and negatives below 0.35. It then fits Platt scalers and evaluates the ensemble on the same rows. That combines target leakage with reuse of calibration data for performance reporting. It does not measure real cloud predictions or independent out-of-fold ensemble accuracy.

The [ensemble code](https://github.com/amnesia-67/sunhacks-2026/blob/6a16734c1af7d4a5ff24400b06849241b285d5fb/src/ensemble.py) uses fixed weights 0.42/0.33/0.25; it does not learn the claimed ridge-regularized stacking weights. The supplied JSON can reproduce that computation, but its availability does not make its probabilities calibrated for a live person. The reported 100% scores therefore should not be presented as live-user accuracy.

## The tabular training indices conflict with the semantic schema

The synthetic training annotations disagree with the committed [250-field schema](https://github.com/amnesia-67/sunhacks-2026/blob/6a16734c1af7d4a5ff24400b06849241b285d5fb/src/feature_schema.py):

|Index|Generator annotation|Actual schema field|
|---:|---|---|
|80|rPPG pNN50|IBI SDNN; pNN50 is index82|
|180|VAD response latency|Praat RAP; VAD latency is index191|
|200|Reality-monitoring sensory ratio|First-person-singular pronoun rate; sensory ratio is index204|

Consequently, supplying 250 real numbers with plausible feature names would not establish semantic compatibility. The Mac's existing seven signal-quality fields must not be zero-padded or renamed into that vector. The schema's indexed eGeMAPS placeholders also need an exact external extractor-field ordering. Branch2 should remain inactive until a valid real-feature training/export contract is supplied and evaluated.

## The source resolves architecture, but not real-input training equivalence

The original [fusion class](https://github.com/amnesia-67/sunhacks-2026/blob/6a16734c1af7d4a5ff24400b06849241b285d5fb/src/multimodal_fusion.py) expects visual `[B,32,1024]`, aligned audio `[B,32,768]`, and text `[B,128]`. It defines the exact Linear/LayerNorm/GELU projectors, DARE temporal salience pooling, discrepancy MLP, reliability softmax and single-output sigmoid heads. These equations are now known from source rather than guessed from tensor shapes.

The [audio wrapper](https://github.com/amnesia-67/sunhacks-2026/blob/6a16734c1af7d4a5ff24400b06849241b285d5fb/src/audio_encoder.py#L98) applies the saved Linear 768/LayerNorm 768/GELU adapter to native Dasheng frame features **before** linear interpolation to 32 steps with `align_corners=False`. It does not merge emotion2vec. The declared 1792-dimensional combined-audio setting is unused in this serving/training route. The Mac's strict compatibility path preserves the actual source architecture instead of loading these weights into the unrelated four-vector reference MLP.

The documented synthetic training route supplies no actual V-JEPA checkpoint/FPS/crop recipe or Qwen instruction/truncation extraction recipe. Using real V-JEPA2.1 tubelet features, real audio and truncated normalized Qwen embeddings can now test integration, but it is outside the synthetic training distribution and does not establish behavioral quality.

## Socket timing claims describe a smaller path

The original [socket loader](https://github.com/amnesia-67/sunhacks-2026/blob/6a16734c1af7d4a5ff24400b06849241b285d5fb/src/v2_socket_server.py#L103) loads the fusion network state but does not load or apply the saved audio adapter. Its `process_jepa` creates random feature tensors when inputs are omitted. The advertised millisecond synthetic result therefore exercises a small fusion head on already feature-shaped inputs, not raw-camera decode, V-JEPA encoding, audio feature extraction or internet transport. The original full Python pipeline does apply the audio adapter, so those serving paths are not equivalent.

The friend's `0.0.0.0:5002` is a bind address, and `127.0.0.1:5002` is that computer's loopback address. Neither provides a reachable authenticated internet endpoint for this Mac. No remote socket connection or participant upload was made by this audit.

## Honest live-demo conclusion

The setup can demonstrate real sensor capture, matching tensor shapes, strict weight loading, finite head outputs and measured per-stage/source-age timings. The live UI must identify these as experimental integration outputs, not validated deception probabilities or reliable judgments about the participant. Keep Branch2 and the leaked ensemble inactive, preserve the original base-model configuration, and require real-data, speaker-disjoint validation plus a matching preprocessing/export contract before making task-performance claims.
