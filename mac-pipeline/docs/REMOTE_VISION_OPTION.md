# Optional RTX 5070 vision service: delayed results with dense motion coverage

This is a vetted design option, not an implemented or configured remote service. The user is considering V-JEPA inference on a friend's RTX5070 over the internet and accepts delayed results if that preserves more motion. No connection to that computer, firewall change, footage upload or remote execution was performed. The selected local contract remains64 frames sampled at24fps; only a later explicit remote configuration would move this branch off the Mac.

## Latency and sustained throughput are different requirements

A fixed delay can be compatible with continuous source-time coverage. For example, a service could return every scheduled window several seconds late while still accepting new frames continuously. It fails to keep up when incoming work arrives faster than upload or GPU processing can sustain; the delay then grows rather than remaining approximately constant. Accepting a longer response delay does not fix that capacity deficit.

The proposed stream sends each timestamped sampled frame once at24fps. The remote host constructs64-frame windows with32-frame hops: a new window every1.333seconds, or0.75windows/second. The first-to-last source timestamp span is2.625seconds. Successful continuous processing requires all scheduled windows to be retained, not merely densely sampled frames inside occasional windows. The number24 describes frames inside the input stream, not24 newly completed model results per second. Eight new window results per second would be a separate0.125-second-hop workload with much heavier overlap.

For one serialized GPU worker, sustained decode/preprocessing/inference service time must be comfortably below1.333seconds per window to retain this schedule. If batching/concurrent execution is used, measure the resulting sustained window throughput and latency together; more outstanding requests do not themselves create GPU capacity. NVIDIA describes batching as a throughput option that also has configurable queue delay, size and timeout controls. These are measurements to make, not speed guarantees for this card. [NVIDIA batching and queue policies](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)

## Persistent upload and independent result handling

Use authenticated TLS, or an authenticated service over an established private VPN. A persistent bidirectional gRPC stream or equivalent framed encrypted connection can upload video while results return independently. Avoid a loop that uploads one whole clip, waits for its inference result and internet round trip, then begins the next upload. That loop artificially ties source throughput to response latency. gRPC's bidirectional streams expressly allow independent reads and writes. Its flow control can block writers and buffer messages, so capture, network sending and result reading must be separate bounded tasks. An application write completing does not prove that its bytes have left the network stack. [gRPC stream semantics](https://grpc.io/docs/what-is-grpc/core-concepts/), [flow control](https://grpc.io/docs/guides/flow-control/)

Bound these separately: the Mac's capture/encode queue in bytes and source seconds; the transport's buffered bytes; the server's received-frame ring; GPU pending windows; and outstanding result identities. Do not use a single one-request credit to throttle the ongoing frame stream until the previous result returns. For a planning latency of `D` seconds and an arrival rate of0.75windows/second, roughly `ceil(0.75 * D)` result identities can be outstanding even when the service is stable. That is a concurrency-planning calculation, not permission for an unbounded GPU queue.

The protocol needs authenticated session identity, frame sequence number, original Mac monotonic timestamp, dimensions/color order, codec settings, preprocessing-stage identifier, contract/model/adapter revisions and per-message size limits. Results echo the session/window IDs and original source start/end timestamps, plus finite1024-wide pooled features and health/timing metadata. Preserve those original timestamps: monotonic clocks on the two computers are not directly comparable. The Mac can calculate total source-age from its own original timestamps when the result arrives; server-local clocks can report server-local stage durations. A delayed result can be valid history and should map to its past timeline interval rather than be relabeled as the current moment. Reject mismatched sessions, duplicates and invalid shapes. Never silently overwrite a newer result with an older one.

## Derive the budget from both upload and compute

Let `S` be measured average encoded bytes per sampled frame, including an allowance for framing, and `U` the measured sustainable application upload bytes/second. The source produces about `24 * S` bytes/second. Require `U > 24 * S` with margin for larger frames, congestion and other traffic. Measure burst sizes and tail delay too; the average alone is insufficient. Illustratively,40KiB per frame would require about7.86Mbps before additional overhead. This is an example, not a predicted codec size.

For the uncompressed384×384RGBuint8 *model-space* example,64frames contain28,311,552bytes, and24frames/second require about84.9Mbps. Re-uploading full50%-overlapping clips approaches twice that steady payload rate. Stream frames once and reuse them in the server ring.

| Sustained upload | Raw64-frame payload transfer minimum |
|---:|---:|
|10Mbps|22.65s|
|20Mbps|11.32s|
|50Mbps|4.53s|
|100Mbps|2.26s|

These calculated minima exclude encoding, propagation, TLS/framing, contention and GPU processing. In a continuous stream, the whole clip is not uploaded again after its final frame is captured: earlier frames are already in transit or buffered remotely. Do not add this whole-clip transfer time to every window as though each were a new independent upload.

Choose a source-end-to-result latency budget `L`. Measure the non-queue delays for encode, final-frame transport, decode/preprocess, GPU service and return. The remainder is a planning budget for network jitter and GPU queueing. Bound queues by bytes, count **and actual source age**; reject expired work using echoed Mac timestamps or explicit relative deadlines. For a conservative per-window service budget `C`, `n` windows ahead can cost about `n * C` waiting time, so pending count must fit the allocated GPU wait budget. Tail measurements are planning evidence, not a hard guarantee of a maximum. NVIDIA's queue controls similarly distinguish maximum count from queue timeout. [Queue policy](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html#queue-policy)

Two explicit policies are reasonable:

- **Coverage within a chosen delay budget:** retain chronological windows while upload and GPU throughput sustain the schedule; a bounded FIFO absorbs short jitter. Deliver delayed results on their original timeline. Expire work only after the stated budget, and count the resulting coverage gaps.
- **Freshest selected clips:** one active GPU task and a latest pending window; replace old pending work under overload. Continue the frame upload independently when its own budget allows. This bounds staleness but deliberately sacrifices temporal coverage and must show skipped intervals.

If sustained GPU service is2seconds for each1.333-second-hop arrival, capacity is0.5windows/second versus0.75arriving. A FIFO grows by roughly0.25windows/second; a longer permitted delay only postpones hitting the limit. If either upload or compute cannot sustain the target, explicitly change hop/sampling/codec settings, drop and report intervals, or stop remote vision. Increasing hop reduces overlap; gaps begin when hop exceeds clip duration. Continuous analysis of every captured instant is not established merely by increasing the camera's requested fps.

## Preprocessing, compression and reconnects

Compression saves bandwidth but can alter motion/detail. Measure actual encoded sizes and recognition/numerical consequences under the teammate's evaluation process; no claim here establishes that a chosen codec preserves subtle cues. Training and inference must match temporal spacing, crop, color conversion and any agreed compression augmentation.

The384px payload arithmetic above does **not** authorize silently sending already cropped384px input through the current adapter unchanged. That adapter always resizes the short side to438 then center-crops384; doing that again would crop a second time. A future transport must explicitly choose either pre-crop capture RGB, with official preprocessing performed once on the server, or model-space cropped RGB with an explicitly implemented preprocessing-stage bypass. Both ends must agree on the contract before teammate tuning.

On reconnect, create a fresh session/stream identity and rebuild enough contiguous timestamped frames to form valid windows; do not join frames across an outage. Independent JPEG/WebP frame messages simplify dropping individual queued frames. Inter-frame codecs require reference/keyframe handling: discarding arbitrary encoded bytes can break decoding, so reset/request a new keyframe and report the discontinuity. Keep local audio, transcript and diagnostic branches usable when remote vision is unavailable. No remote or mock verdict should be invented during an outage.

## First benchmark the GPU without a camera or network service

NVIDIA lists RTX5070 as a12GB GDDR7 Blackwell card with CUDA compute capability12.0; ensure its driver/PyTorch build actually supports the card. Specifications and AI TOPS cannot predict this model's throughput. [Official RTX5070 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/)

On that computer, obtain the same official checkpoint and reviewed source revision, use a working compatible CUDA PyTorch environment, and configure `models.vision.device="cuda"` with `dtype="float16"`. Run:

```sh
python -m sunhacks benchmark --branch vision --frames 64 --repeats 5 --config gpu-config.json --output gpu-benchmark.json
```

The current helper accepts `cuda`, checks availability, uses CUDA autocast for reduced precision and synchronizes CUDA around the timed work. Independent dispatch testing with a fake backend confirmed that `_sync(...,"cuda")` calls CUDA synchronization and CPU mode does not. No real CUDA execution was possible on this Mac. PyTorch documents that synchronization waits for kernels on the selected/current CUDA device, making the wall-clock timing meaningful. [CUDA synchronization](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.synchronize.html)

The helper reports preprocessing+inference latency after a separately timed load, and CUDA peak allocated/reserved memory since process start; use a fresh process for interpretable peaks. It currently supports the default CUDA device string, not an indexed multi-GPU selector. This smoke test is a prerequisite, not the streaming acceptance test. Record GPU name, driver/CUDA/PyTorch versions, exact model/source revisions, dtype/autocast and competing workloads with the report.

Then measure the complete prototype: sustained upload, encoded bytes/frame and bursts, decode/preprocess throughput, sustained GPU windows/second, latency distribution, time-budget expirations, source-time coverage, memory and outage/reconnect behavior. Accepting several seconds of stable delay is compatible with full scheduled coverage **if these capacities sustain the stream**. Neither the internet path nor the RTX5070 performance has been measured here; no speed promise is made.
