"""Bounded VIDEO-ONLY adapter for reviewed 488a818 ViT-L jepa_video classifier.

No capture, downloads, model execution, networking on import/construction,
legacy V-JEPA2 fallback, analyze endpoint, audio or transcript transmission.
"""
from __future__ import annotations

import ipaddress
import json
import math
import socket
import threading
import time
import uuid

SOURCE_REVISION = "488a818864684b0e0b2a2275866e996e06eaef5c"
MODEL_LABEL = "vjepa2_1_vitl_lora_5fold"
MODEL_ID = "facebookresearch/vjepa2:vjepa2_1_vit_large_384"
MANIFEST_SHA256 = "30d1f7fc8dcce8307a0193bcaab5b345500e7cc4cb6f912c1fefdaf8f8012b32"
SOURCE_TREE_SHA256 = "e9eb35d81ffa524462a5f14c5583aba07a6dce2049cba607fc4868aa9ab45acb"
# Independently reconstructed from the reviewed commit's 55 source files plus
# the exact unused eval draft published in handoff commit 587035e. Serving
# modules are unchanged. Keep an exact allowlist; never accept arbitrary dirty
# trees or infer content identity from the reported commit/dirty flag.
# Evidence: reports/served_5002/record.json and its untracked_at_start copy at
# https://github.com/amnesia-67/sunhacks-2026/tree/587035efc41ee741c8bc4b1c0c25e7d35c645420
SOURCE_TREE_SHA256_WITH_EVAL_DRAFT = "296402ed39632b90e2ca6643de3e0a3a096a07c46fd7ce556a4e74d4b208e7c8"
# 70cfe69 publishes the provenance-only repair: tracked source hashing,
# normalized JSON hashes, and revision metadata in ping. Reconstructed from
# 58 published source files; the L model and video processing are unchanged.
SOURCE_TREE_SHA256_PROVENANCE_REPAIR = "40cf10467455baa732edfe67cc54fe0ef960229a0ed22ff2a1ca1cf4c28b738a"
REVIEWED_SOURCE_TREES = frozenset((SOURCE_TREE_SHA256, SOURCE_TREE_SHA256_WITH_EVAL_DRAFT, SOURCE_TREE_SHA256_PROVENANCE_REPAIR))

CONTRACT = {'backbone': 'vjepa2_1_l', 'cache_frames': 128, 'cache_window_frames': 64, 'classifier_only': True, 'encoder': {'checkpoint_key': 'ema_encoder', 'depth': 24, 'embed_dim': 1024, 'heads': 16, 'patch_size': 16, 'tubelet_size': 2, 'upstream_commit': '204698b45b3712590f06245fbfba32d3be539812'}, 'fold_aggregation': 'mean of the 5 fold logits, each the mean over the 2 views', 'frame_stride': 1, 'frames_per_view': 64, 'hidden_width': 1024, 'model_id': 'facebookresearch/vjepa2:vjepa2_1_vit_large_384', 'native_temporal_tubelets': 32, 'official_profile': 'Meta hub builder vjepa2_1_vit_large_384: num_frames=64, tubelet_size=2, img_size=384, patch_size=16, use_rope=True, interpolate_rope=True; encoder built with exactly these', 'preprocessing': 'rlt-face384-square-bilinear-antialias-div255-imagenet-v1', 'sampling': 'time_uniform_128_nearest_frame; timestamps or fps required; blazeface crop on node 2', 'spatial_pooling': 'mean_over_24x24_patches', 'spatial_resolution': 384, 'temporal_interpolation': 'none (32 native tubelets)', 'temporal_steps': 32, 'variant': 'ViT-L', 'views': 2}
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024


class RemoteClassifierError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite value")
    if isinstance(value, dict):
        for item in value.values(): _finite(item)
    elif isinstance(value, list):
        for item in value: _finite(item)


def _number(value, *, low=0, high=None):
    try:
        return (not isinstance(value, bool) and isinstance(value, (int, float)) and
                math.isfinite(value) and value >= low and (high is None or value <= high))
    except OverflowError:
        return False


# The requested deployment branch adds a single-model pilot profile. The
# running server's 61 tracked source files were reconstructed at 7426904.
PILOT_MANIFEST_SHA256="e0910ae6d979521db7f2af25a91d22647c0be87a462cd0ff5af9f686a5fdb9f8"
PILOT_MODEL_LABEL="vjepa2_1_vitl_lora_webcam_pilot"
PILOT_CONTRACT={**CONTRACT,"fold_aggregation":"single model (no folds): mean of the 2 view logits"}
REVIEWED_SOURCE_TREES=REVIEWED_SOURCE_TREES | {"1440d8435a07666eac67ea4e5ca330b2e1d043815229eeb6b877245c11431f8c","cc1d674b8d1b13bf652bc9a82e5d9c24b5a49506233f0ed9e7c078fc9d24d1d8","4457932e7e407a61620a2645155ffddf5f4414c4da7449292600022877193726"}

class RemoteJepa21LClassifier:
    def __init__(self, host, port=5002, *, reviewed_source_revision=None,
                 expected_manifest_sha256=MANIFEST_SHA256,
                 participant_media_authorized=False, timeout_seconds=15, max_frame_edge=640, profile="served"):
        if type(max_frame_edge) is not int or not 64 <= max_frame_edge <= 960:
            raise ValueError("Transport max_frame_edge must be an integer within64..960")
        address = ipaddress.ip_address(host)
        if address.is_unspecified or address.is_multicast or str(address) == "255.255.255.255":
            raise ValueError("Use a specific verified peer IP")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid port")
        if not _number(timeout_seconds, low=.01, high=60):
            raise ValueError("Total deadline must be within0.01..60seconds")
        if type(participant_media_authorized) is not bool:
            raise ValueError("participant_media_authorized must be boolean")
        if profile not in ("served","webcam_pilot"):raise ValueError("Unknown video model profile")
        self._profile=profile
        self._model_label=PILOT_MODEL_LABEL if profile=="webcam_pilot" else MODEL_LABEL
        self._contract=PILOT_CONTRACT if profile=="webcam_pilot" else CONTRACT
        self._fold_count=1 if profile=="webcam_pilot" else 5
        self._manifest_artifact="models/pilot_webcam/vjepa2_1_l/deploy_meta.json" if profile=="webcam_pilot" else "models/vjepa2_1_l_lora/deploy_meta.json"
        if expected_manifest_sha256 != (PILOT_MANIFEST_SHA256 if profile=="webcam_pilot" else MANIFEST_SHA256):
            raise ValueError("This adapter supports only the exact reviewed2.1L manifest")
        self._endpoint = (str(address), port, address.version)
        self._reviewed_source = reviewed_source_revision
        self._expected_manifest = expected_manifest_sha256
        self._participant_authorized = participant_media_authorized
        self._timeout = float(timeout_seconds)
        self._max_frame_edge = max_frame_edge
        self._max_source_bytes = (256 if max_frame_edge <= 640 else 384) * 1024 * 1024
        self._flight = threading.Lock()
        self._state_lock = threading.Lock()
        self._health_ready = False
        self._closed = False
        self._active_socket = None
        self._state = {"status": "not_probed", "busy": False, "classifier_only": True,
                       "model_id": MODEL_ID, "participant_media_authorized": participant_media_authorized,
                       "transport_max_frame_edge":max_frame_edge,"transport_codec":"jpeg",
                       "transport_jpeg_quality":90,"transport_jpeg_subsampling":"4:4:4"}

    def _update(self, **changes):
        with self._state_lock:
            if self._closed:
                for field in ("status","classifier_ready","classifier_response_validated"):
                    changes.pop(field,None)
            self._state.update(changes)

    def _check_open(self):
        with self._state_lock:
            if self._closed:
                raise RemoteClassifierError("closed", "Client closed; no further transmission/result is permitted")

    def _publish_open(self, changes):
        with self._state_lock:
            if self._closed:
                raise RemoteClassifierError("closed", "Client closed before publication; discarding late score")
            self._state.update(changes)

    def snapshot(self):
        with self._state_lock:
            return dict(self._state)

    def close(self):
        with self._state_lock:
            self._closed = True
            self._health_ready = False
            stream = self._active_socket
            self._state.update(status="closed",classifier_ready=False,classifier_response_validated=False)
        if stream is not None:
            try: stream.shutdown(socket.SHUT_RDWR)
            except (OSError,AttributeError): pass
            try: stream.close()
            except (OSError,AttributeError): pass

    @staticmethod
    def _remaining(deadline):
        left = deadline - time.monotonic()
        if left <= 0:
            raise RemoteClassifierError("timeout", "Total classifier request deadline exceeded")
        return left

    def _exchange(self, payload, deadline, maximum=MAX_RESPONSE_BYTES):
        self._check_open()
        if self._profile=="webcam_pilot" and payload.get("action")=="jepa_video":payload={**payload,"profile":"webcam_pilot"}
        packed = _canonical(payload).encode("utf-8") + b"\n"
        if len(packed) > MAX_REQUEST_BYTES:
            raise RemoteClassifierError("request_too_large", "Video request exceeds16MiB")
        host, port, version = self._endpoint
        try:
            family = socket.AF_INET6 if version == 6 else socket.AF_INET
            with socket.socket(family, socket.SOCK_STREAM) as stream:
                with self._state_lock:
                    if self._closed:
                        raise RemoteClassifierError("closed", "Client closed before connection")
                    self._active_socket = stream
                self._check_open()
                stream.settimeout(self._remaining(deadline))
                stream.connect((host, port))
                stream.settimeout(self._remaining(deadline))
                self._check_open()
                stream.sendall(packed)
                buffer = bytearray()
                while b"\n" not in buffer:
                    self._check_open()
                    stream.settimeout(self._remaining(deadline))
                    chunk = stream.recv(min(16384, maximum + 1 - len(buffer)))
                    if not chunk:
                        raise RemoteClassifierError("protocol_error", "Peer closed before NDJSON delimiter")
                    buffer.extend(chunk)
                    if len(buffer) > maximum:
                        raise RemoteClassifierError("protocol_error", "Response exceeds configured byte limit")
                line, _, tail = bytes(buffer).partition(b"\n")
                if tail.strip():
                    raise RemoteClassifierError("protocol_error", "Expected one response object")
                response = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs,
                                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
                _finite(response)
                self._remaining(deadline)
                self._check_open()
                if not isinstance(response, dict):
                    raise ValueError("response must be object")
                return response, len(packed), len(buffer)
        except RemoteClassifierError:
            raise
        except (socket.timeout, TimeoutError):
            raise RemoteClassifierError("timeout", "Socket exceeded total request deadline") from None
        except OSError as exc:
            if self._closed:
                raise RemoteClassifierError("closed", "Client closed during socket operation") from None
            raise RemoteClassifierError("network_error", type(exc).__name__ + (" errno="+str(exc.errno) if exc.errno is not None else "")) from None
        except (UnicodeError, ValueError, OverflowError, RecursionError):
            raise RemoteClassifierError("protocol_error", "Invalid bounded JSON response") from None
        finally:
            with self._state_lock:
                self._active_socket = None

    def probe(self):
        """Explicit metadata-only ping. No automatic probe occurs on predict."""
        if self._closed:
            return {"status": "closed", "classifier_ready": False}
        if not self._flight.acquire(blocking=False):
            return {"status": "busy", "classifier_ready": False}
        self._health_ready = False
        self._update(classifier_ready=False,classifier_response_validated=False)
        started = time.monotonic()
        try:
            health, _, _ = self._exchange({"action": "ping"}, started + min(3,self._timeout), 65536)
            encoder = (health.get("profiles") or {}).get("webcam_pilot") if self._profile=="webcam_pilot" else health.get("jepa")
            if (health.get("status") != "ok" or health.get("jepa_loaded") is not True or
                    not isinstance(encoder, dict) or encoder.get("model") != self._model_label or
                    encoder.get("model_id") != MODEL_ID or type(encoder.get("folds")) is not int or
                    encoder["folds"] != self._fold_count or encoder.get("manifest_sha256") != self._expected_manifest or
                    _canonical(encoder.get("contract")) != _canonical(self._contract)):
                raise RemoteClassifierError("identity_mismatch", "Peer does not advertise the exact reviewed V-JEPA2.1 ViT-L classifier")
            if encoder.get("device") not in ("cuda", "cuda:0"):
                raise RemoteClassifierError("identity_mismatch", "Expected the declared CUDA video classifier")
            self._check_open()
            value = {"status": "classifier_contract_declared", "classifier_ready": True,
                     "classifier_response_validated": False, "model_id": MODEL_ID,
                     "manifest_sha256": self._expected_manifest,
                     "elapsed_ms": round((time.monotonic()-started)*1000,3)}
            self._publish_open(value)
            with self._state_lock:
                if self._closed:
                    raise RemoteClassifierError("closed", "Client closed during metadata probe")
                self._health_ready = True
            return value
        except RemoteClassifierError as exc:
            value = {"status": exc.code, "classifier_ready": False, "reason": str(exc)}
            self._update(**value)
            return value
        finally:
            self._flight.release()

    def predict(self, clip, *, window_id, timestamps, face_crop=True):
        """Participant video requires separate authorization; no audio/text API exists."""
        if not self._participant_authorized:
            raise RemoteClassifierError("media_authorization_required", "Participant video sending is disabled until the live test is authorized")
        return self._predict(clip,window_id=window_id,timestamps=timestamps,face_crop=face_crop,fixture=False)

    def benchmark_synthetic(self, *, variant=0, window_id=0):
        """Internally generated non-person pixels only; never accepts caller media."""
        import numpy as np
        if type(variant) is not int or variant not in (0,1):
            raise ValueError("Synthetic variant must be0or1")
        yy,xx = np.indices((128,128))
        clip = np.zeros((128,128,128,3),dtype=np.uint8)
        for index in range(128):
            board = ((xx//16 + yy//16 + variant)%2)*70
            clip[index,:,:,0] = board+20
            clip[index,:,:,1] = 30 + variant*40
            left = (index*3 + variant*19)%100
            clip[index,30:80,left:left+24,2] = 220
        stamps = np.arange(128,dtype=np.float64)/24
        return self._predict(clip,window_id=window_id,timestamps=stamps,face_crop=False,fixture=True)

    def _predict(self, clip, *, window_id, timestamps, face_crop, fixture):
        if self._closed:
            raise RemoteClassifierError("closed", "Client is closed")
        if self._reviewed_source != SOURCE_REVISION:
            raise RemoteClassifierError("review_required", "Exact pushed source must be reviewed before video requests")
        if not self._health_ready:
            raise RemoteClassifierError("probe_required", "Run an explicit matching metadata probe before video")
        import base64
        import io
        import numpy as np
        from PIL import Image
        if not self._flight.acquire(blocking=False):
            raise RemoteClassifierError("busy", "Previous video request is still active; this window was not queued")
        started = time.monotonic()
        deadline = started + self._timeout
        self._update(busy=True,status="packing")
        try:
            if (not isinstance(clip,np.ndarray) or clip.dtype != np.uint8 or clip.ndim != 4 or
                    not 16 <= clip.shape[0] <= 900 or clip.shape[-1] != 3 or min(clip.shape[1:3]) < 1 or
                    clip.nbytes > self._max_source_bytes):
                raise RemoteClassifierError("input_error", "Requires16..900 RGB uint8 frames shaped[N,H,W,3], within the bounded source-memory limit")
            if type(window_id) is not int or not 0 <= window_id <= 2**53-1 or type(face_crop) is not bool:
                raise RemoteClassifierError("input_error", "Invalid source window or face_crop flag")
            self._update(source_window_id=window_id)
            raw_stamps = np.asarray(timestamps)
            if (raw_stamps.ndim != 1 or len(raw_stamps) != len(clip) or raw_stamps.dtype.kind not in "iuf" or
                    not np.isfinite(raw_stamps).all()):
                raise RemoteClassifierError("input_error", "One finite numeric timestamp per source frame is required")
            stamps = raw_stamps.astype(np.float64)
            if not np.isfinite(stamps).all() or np.any(stamps < 0) or np.any(np.diff(stamps) <= 0) or stamps[-1] <= stamps[0]:
                raise RemoteClassifierError("input_error", "Raw capture timestamps must strictly increase with positive total span")
            from .local_digest import select_timestamp_frames
            relative_stamps=stamps-stamps[0]
            cache_indices,_=select_timestamp_frames(relative_stamps)
            # Retain all model-selected images plus both time endpoints. The server
            # then selects exactly the same cache once, without encoding unused images.
            transport_indices=np.unique(np.r_[0,cache_indices,len(clip)-1])
            height,width = clip.shape[1:3]
            scale=min(1.,self._max_frame_edge/max(height,width))
            size=(max(1,round(width*scale)),max(1,round(height*scale)))
            request_id=uuid.uuid4().hex
            payload={"action":"jepa_video","frames":[""]*len(transport_indices),
                     "timestamps":relative_stamps[transport_indices].tolist(),
                     "request_id":request_id,"source_id":window_id,"client_revision":SOURCE_REVISION}
            if self._profile=="webcam_pilot":payload["profile"]="webcam_pilot"
            # Budget the actual NDJSON envelope too, not just JPEG bytes. Four
            # representative frames pick a likely fitting quality/size before
            # encoding the entire clip; a bounded retry handles uneven detail.
            image_budget=MAX_REQUEST_BYTES-len(_canonical(payload).encode("utf-8"))-1
            if image_budget<=0:raise RemoteClassifierError("request_too_large","Video metadata exceeds request limit")
            quality=90
            def encode(index):
                self._check_open();self._remaining(deadline)
                image=Image.fromarray(clip[index])
                if image.size!=size:image=image.resize(size,Image.Resampling.LANCZOS)
                output=io.BytesIO();image.save(output,format="JPEG",quality=quality,subsampling=0,optimize=False)
                return base64.b64encode(output.getvalue()).decode("ascii")
            sample_positions=np.unique(np.linspace(0,len(transport_indices)-1,min(4,len(transport_indices)),dtype=int))
            estimate=max(len(encode(transport_indices[i])) for i in sample_positions)*len(transport_indices)
            if estimate>image_budget*.94:
                quality=80
                estimate=max(len(encode(transport_indices[i])) for i in sample_positions)*len(transport_indices)
            if estimate>image_budget*.94:
                ratio=min(.9,(image_budget*.85/estimate)**.5)
                size=(max(1,int(size[0]*ratio)),max(1,int(size[1]*ratio)))
            for attempt in range(5):
                encoded=[];encoded_bytes=0
                for index in transport_indices:
                    value=encode(index);encoded.append(value);encoded_bytes+=len(value)
                    if encoded_bytes>image_budget:break
                if encoded_bytes<=image_budget and len(encoded)==len(transport_indices):break
                # Uniform resize preserves the whole selected timeline and
                # identical geometry across frames. Never drop frames to fit.
                expected=encoded_bytes*len(transport_indices)/len(encoded)
                ratio=min(.8,(image_budget*.8/expected)**.5)
                next_size=(max(1,int(size[0]*ratio)),max(1,int(size[1]*ratio)))
                if next_size==size or attempt==4:
                    raise RemoteClassifierError("request_too_large","Video cannot fit the socket request budget")
                size=next_size
            payload["frames"]=encoded
            packing_ms=(time.monotonic()-started)*1000
            self._update(status="waiting_for_classifier")
            response,request_bytes,response_bytes=self._exchange(
                payload,deadline)
            branch=self._validate(response,len(encoded),request_id=request_id,window_id=window_id)
            if self._closed:
                raise RemoteClassifierError("closed", "Client closed before result; discarding late score")
            self._remaining(deadline)
            # Match the server's single nearest-time selection on ORIGINAL capture frames.
            from .local_digest import select_timestamp_frames
            selected,_=select_timestamp_frames(relative_stamps)
            consumed=len(set(selected.tolist()))
            face=response.get("face")
            if face is not None and (type(face.get("unique_frames_used")) is not int or face["unique_frames_used"]!=consumed):
                raise RemoteClassifierError("protocol_error", "Server temporal sampling differs from the source timestamps")
            branch.update(source_window_id=window_id,request_id=request_id,
                _source_frame_timestamps=tuple(float(value) for value in stamps),
                source_start=float(stamps[0]),source_end=float(stamps[-1]),
                source_frames_sent=len(encoded),source_unique_timestamps=len(stamps),
                raw_source_frames=len(clip),transport_selection="nearest_time128_plus_original_endpoints",
                encoder_unique_source_frames=consumed,
                server_echoes_source_identity=True,source_alignment="request_id_source_id_client_revision_and_capture_timestamps",
                synthetic_fixture=fixture,face_crop_requested=face_crop,
                source_frame_shape=[int(height),int(width),3],transport_frame_shape=[int(size[1]),int(size[0]),3],
                input_rgb_bytes=int(clip.nbytes),transport_max_frame_edge=self._max_frame_edge,
                transport_resized=(int(width),int(height))!=size,transport_codec="jpeg",
                transport_jpeg_quality=quality,transport_jpeg_subsampling="4:4:4",transport_jpeg_huffman_optimized=False,
                transport_budget_adapted=(quality!=90 or size!=(max(1,round(width*scale)),max(1,round(height*scale)))),
                transport_request_limit_bytes=MAX_REQUEST_BYTES,
                elapsed_ms=round((time.monotonic()-started)*1000,3),client_packing_ms=round(packing_ms,3),
                request_bytes=request_bytes,response_bytes=response_bytes)
            self._publish_open({"status":"ready","last_elapsed_ms":branch["elapsed_ms"],
                                "last_window_id":window_id,"classifier_response_validated":True,
                                "transport_jpeg_quality":quality,"transport_frame_shape":[size[1],size[0],3],
                                "last_request_bytes":request_bytes})
            return branch
        except RemoteClassifierError as exc:
            if exc.code not in ("input_error","request_too_large"):
                self._health_ready=False
                self._update(classifier_ready=False,classifier_response_validated=False)
            self._update(status=exc.code,last_error=str(exc))
            raise
        except (ValueError,TypeError,OverflowError):
            self._update(status="input_error")
            raise RemoteClassifierError("input_error", "Invalid video/timestamp packing input") from None
        finally:
            self._update(busy=False)
            self._flight.release()

    def _validate(self,response,frame_count,*,request_id,window_id):
        if response.get("status") != "success":
            raise RemoteClassifierError("server_error", "Video classifier returned an error")
        if (response.get("model") != self._model_label or response.get("model_id") != MODEL_ID or
                response.get("manifest_sha256") != self._expected_manifest or
                response.get("classifier_only") is not True or (response.get("calibrated") is not False if self._profile=="served" else response.get("calibrated")!="platt_on_pilot_validation") or
                response.get("device") not in ("cuda","cuda:0") or
                _canonical(response.get("contract")) != _canonical(self._contract) or
                type(response.get("n_frames_received")) is not int or response["n_frames_received"] != frame_count):
            raise RemoteClassifierError("identity_mismatch", "Video response differs from reviewed2.1L classifier/manifest/shape contract")
        if (response.get("request_id")!=request_id or type(response.get("source_id")) is not int or
                response["source_id"]!=window_id or response.get("client_revision")!=SOURCE_REVISION):
            raise RemoteClassifierError("identity_mismatch", "Reply does not echo this exact video request and source")
        revision=response.get("server_revision")
        code=revision.get("code") if isinstance(revision,dict) else None
        artifacts=revision.get("artifacts_sha256") if isinstance(revision,dict) else None
        if (not isinstance(code,dict) or code.get("source_tree_sha256") not in REVIEWED_SOURCE_TREES or
                not isinstance(artifacts,dict) or artifacts.get(self._manifest_artifact)!=self._expected_manifest):
            raise RemoteClassifierError("identity_mismatch", "Server does not declare the reviewed source bytes and L manifest")
        score=response.get("raw_score")
        folds=response.get("per_fold_probability")
        attention=response.get("temporal_attention_32")
        if (not _number(score,high=1) or not isinstance(folds,list) or len(folds)!=self._fold_count or
                not all(_number(value,high=1) for value in folds) or
                not isinstance(attention,list) or len(attention)!=32 or
                not all(_number(value,high=1) for value in attention) or abs(sum(attention)-1)>.001 or
                not _number(response.get("prep_ms")) or not _number(response.get("inference_ms")) or
                response["inference_ms"] <= 0):
            raise RemoteClassifierError("protocol_error", "Invalid finite classifier scores, attention or timings")
        if self._profile=="webcam_pilot" and (response.get("profile")!="webcam_pilot" or not _number(response.get("calibrated_probability"),high=1) or not _number(response.get("threshold"),high=1)):
            raise RemoteClassifierError("identity_mismatch","Invalid pilot profile/calibration metadata")
        face=response.get("face")
        if face is not None and (not isinstance(face,dict) or
                not _number(face.get("face_detect_rate"),high=1) or
                type(face.get("centre_crop_fallback")) is not bool):
            raise RemoteClassifierError("protocol_error", "Invalid remote face-crop diagnostics")
        output={"status":"trained_classifier","model":self._model_label,"profile":self._profile,"model_id":MODEL_ID,
                "manifest_sha256":self._expected_manifest,"reviewed_source_revision":SOURCE_REVISION,
                "remote_source_revision_verified":False,"server_declared_source_tree_sha256":code["source_tree_sha256"],
                "provenance_scope":"server-reported metadata; not independent remote attestation","device":response["device"],
                "classifier_only":True,"calibrated":False,"raw_score":float(score),
                "per_fold_scores":[float(value) for value in folds],
                "temporal_attention_32":[float(value) for value in attention],
                "attention_alignment":"within_view_bins_averaged_across_two_halves_and_"+str(self._fold_count)+"_models",
                "server_timings":{"prep_ms":float(response["prep_ms"]),"inference_ms":float(response["inference_ms"])},
                "embedding_shape":None,"input_contract":"sunhacks-vjepa21-vitl-classifier-v1",
                "reason":"Real tuned2.1ViT-L classifier score; uncalibrated for webcam use and not evidence of deception."}
        if face is not None:
            output["face"]={"face_detect_rate":float(face["face_detect_rate"]),
                            "centre_crop_fallback":face["centre_crop_fallback"]}
        if self._profile=="webcam_pilot":
            output.update(calibrated_probability=response["calibrated_probability"],threshold=response["threshold"],calibration_scope="pilot_validation_only",metrics=response.get("metrics"),caveats=response.get("caveats",[]))
        return output
