"""Bounded VIDEO-ONLY adapter for reviewed 5339358 jepa_video classifier.

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

SOURCE_REVISION = "53393582c2994025f2bbececc01eb7a893de1c96"
MODEL_LABEL = "vjepa2_1_vitb_lora_5fold"
MODEL_ID = "facebookresearch/vjepa2:vjepa2_1_vit_base_384"
MANIFEST_SHA256 = "d93d19e751461e4fb906ea32ddaf0a3d0c5ad774585c27103ebd1b42b64cf03d"
CONTRACT = {
    "backbone": "vjepa2_1", "model_id": MODEL_ID, "variant": "ViT-B",
    "classifier_only": True, "cache_frames": 128, "views": 2,
    "cache_window_frames": 64, "frames_per_view": 32, "frame_stride": 2,
    "spatial_resolution": 384, "hidden_width": 768,
    "native_temporal_tubelets": 16, "interpolated_temporal_steps": 32,
    "temporal_interpolation": "linear_align_corners_false",
    "preprocessing": "rlt-face384-square-bilinear-antialias-div255-imagenet-v1",
    "sampling_limitation": "count_uniform_to128_repeats_short_inputs_no_timestamps",
}
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


class RemoteJepa21Classifier:
    def __init__(self, host, port=5002, *, reviewed_source_revision=None,
                 expected_manifest_sha256=MANIFEST_SHA256,
                 participant_media_authorized=False, timeout_seconds=15, max_frame_edge=640):
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
        if expected_manifest_sha256 != MANIFEST_SHA256:
            raise ValueError("This adapter supports only the exact reviewed2.1B manifest")
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
            encoder = health.get("jepa")
            if (health.get("status") != "ok" or health.get("jepa_loaded") is not True or
                    not isinstance(encoder, dict) or encoder.get("model") != MODEL_LABEL or
                    encoder.get("model_id") != MODEL_ID or type(encoder.get("folds")) is not int or
                    encoder["folds"] != 5 or encoder.get("manifest_sha256") != self._expected_manifest or
                    _canonical(encoder.get("contract")) != _canonical(CONTRACT)):
                raise RemoteClassifierError("identity_mismatch", "Peer does not advertise the exact reviewed V-JEPA2.1 ViT-B classifier")
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
                    not 16 <= clip.shape[0] <= 128 or clip.shape[-1] != 3 or min(clip.shape[1:3]) < 1 or
                    clip.nbytes > self._max_source_bytes):
                raise RemoteClassifierError("input_error", "Requires16..128 RGB uint8 frames shaped[N,H,W,3], within the bounded source-memory limit")
            if type(window_id) is not int or not 0 <= window_id <= 2**53-1 or type(face_crop) is not bool:
                raise RemoteClassifierError("input_error", "Invalid source window or face_crop flag")
            self._update(source_window_id=window_id)
            raw_stamps = np.asarray(timestamps)
            if (raw_stamps.ndim != 1 or len(raw_stamps) != len(clip) or raw_stamps.dtype.kind not in "iuf" or
                    not np.isfinite(raw_stamps).all()):
                raise RemoteClassifierError("input_error", "One finite numeric timestamp per source frame is required")
            stamps = raw_stamps.astype(np.float64)
            if not np.isfinite(stamps).all() or np.any(stamps < 0) or np.any(np.diff(stamps) < 0) or stamps[-1] <= stamps[0]:
                raise RemoteClassifierError("input_error", "Selected timestamps must be nondecreasing with positive total span; duplicate selections remain explicit")
            height,width = clip.shape[1:3]
            scale=min(1.,self._max_frame_edge/max(height,width))
            size=(max(1,round(width*scale)),max(1,round(height*scale)))
            encoded=[]
            encoded_bytes=0
            for frame in clip:
                self._check_open()
                self._remaining(deadline)
                image=Image.fromarray(frame)
                if image.size != size: image=image.resize(size,Image.Resampling.LANCZOS)
                output=io.BytesIO()
                image.save(output,format="JPEG",quality=90,subsampling=0,optimize=True)
                value=base64.b64encode(output.getvalue()).decode("ascii")
                encoded_bytes+=len(value)
                if encoded_bytes > MAX_REQUEST_BYTES:
                    raise RemoteClassifierError("request_too_large", "Packed frames exceed16MiB")
                encoded.append(value)
            packing_ms=(time.monotonic()-started)*1000
            self._update(status="waiting_for_classifier")
            response,request_bytes,response_bytes=self._exchange(
                {"action":"jepa_video","frames":encoded,"face_crop":face_crop},deadline)
            branch=self._validate(response,len(clip))
            if self._closed:
                raise RemoteClassifierError("closed", "Client closed before result; discarding late score")
            self._remaining(deadline)
            #Server ignores capture timestamps: keep exact source metadata local.
            selected=np.floor((np.arange(128)+.5)*len(clip)/128).astype(int)
            native_positions=np.concatenate([np.arange(0,64,2),np.arange(64,128,2)])
            branch.update(source_window_id=window_id,request_id=str(uuid.uuid4()),
                _source_frame_timestamps=tuple(float(value) for value in stamps),
                source_start=float(stamps[0]),source_end=float(stamps[-1]),
                source_frames_sent=len(clip),source_unique_timestamps=len(set(float(stamp) for stamp in stamps)),
                encoder_unique_source_frames=len(set(float(stamps[index]) for index in selected[native_positions])),
                server_echoes_source_identity=False,source_alignment="local_request_connection_and_timestamps_only",
                synthetic_fixture=fixture,face_crop_requested=face_crop,
                source_frame_shape=[int(height),int(width),3],transport_frame_shape=[int(size[1]),int(size[0]),3],
                input_rgb_bytes=int(clip.nbytes),transport_max_frame_edge=self._max_frame_edge,
                transport_resized=(int(width),int(height))!=size,transport_codec="jpeg",
                transport_jpeg_quality=90,transport_jpeg_subsampling="4:4:4",transport_jpeg_huffman_optimized=True,
                elapsed_ms=round((time.monotonic()-started)*1000,3),client_packing_ms=round(packing_ms,3),
                request_bytes=request_bytes,response_bytes=response_bytes)
            self._publish_open({"status":"ready","last_elapsed_ms":branch["elapsed_ms"],
                                "last_window_id":window_id,"classifier_response_validated":True})
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

    def _validate(self,response,frame_count):
        if response.get("status") != "success":
            raise RemoteClassifierError("server_error", "Video classifier returned an error")
        if (response.get("model") != MODEL_LABEL or response.get("model_id") != MODEL_ID or
                response.get("manifest_sha256") != self._expected_manifest or
                response.get("classifier_only") is not True or response.get("calibrated") is not False or
                response.get("device") not in ("cuda","cuda:0") or
                _canonical(response.get("contract")) != _canonical(CONTRACT) or
                type(response.get("n_frames_received")) is not int or response["n_frames_received"] != frame_count):
            raise RemoteClassifierError("identity_mismatch", "Video response differs from reviewed2.1B classifier/manifest/shape contract")
        score=response.get("raw_score")
        folds=response.get("per_fold_probability")
        attention=response.get("temporal_attention_32")
        if (not _number(score,high=1) or not isinstance(folds,list) or len(folds)!=5 or
                not all(_number(value,high=1) for value in folds) or
                not isinstance(attention,list) or len(attention)!=32 or
                not all(_number(value,high=1) for value in attention) or abs(sum(attention)-1)>.001 or
                not _number(response.get("prep_ms")) or not _number(response.get("inference_ms")) or
                response["inference_ms"] <= 0):
            raise RemoteClassifierError("protocol_error", "Invalid finite classifier scores, attention or timings")
        face=response.get("face")
        if face is not None and (not isinstance(face,dict) or
                not _number(face.get("face_detect_rate"),high=1) or
                type(face.get("centre_crop_fallback")) is not bool):
            raise RemoteClassifierError("protocol_error", "Invalid remote face-crop diagnostics")
        output={"status":"trained_classifier","model":MODEL_LABEL,"model_id":MODEL_ID,
                "manifest_sha256":self._expected_manifest,"reviewed_source_revision":SOURCE_REVISION,
                "remote_source_revision_verified":False,"device":response["device"],
                "classifier_only":True,"calibrated":False,"raw_score":float(score),
                "per_fold_scores":[float(value) for value in folds],
                "temporal_attention_32":[float(value) for value in attention],
                "attention_alignment":"within_view_bins_averaged_across_two_halves_and_five_folds",
                "server_timings":{"prep_ms":float(response["prep_ms"]),"inference_ms":float(response["inference_ms"])},
                "embedding_shape":None,"input_contract":"sunhacks-vjepa21-vitb-classifier-v1",
                "reason":"Real tuned2.1ViT-B classifier score; uncalibrated for webcam use and not evidence of deception."}
        if face is not None:
            output["face"]={"face_detect_rate":float(face["face_detect_rate"]),
                            "centre_crop_fallback":face["centre_crop_fallback"]}
        return output
