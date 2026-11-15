"""Prepared bounded health and raw-video clients for a reviewed peer service.

Construction/import performs no I/O. Explicit raw-video calls require reviewed
pins and matching health; no fallback to the legacy feature-input head exists.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import socket
import time

PING = b'{"action":"ping"}\n'


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite health value")
    if isinstance(value, dict):
        for item in value.values(): _finite_json(item)
    elif isinstance(value, list):
        for item in value: _finite_json(item)


def _pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output: raise ValueError("Duplicate health field")
        output[key] = value
    return output


def _safe_metadata(value, depth=0):
    if depth > 5: return "[nested metadata omitted]"
    if isinstance(value, str): return value[:4096]
    if isinstance(value, list): return [_safe_metadata(x, depth + 1) for x in value[:32]]
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:64]:
            if isinstance(item, str) and any(word in key.lower() for word in ("secret", "password", "token", "api_key", "credential")):
                result[key[:100]] = "[redacted]"
            else: result[key[:100]] = _safe_metadata(item, depth + 1)
        return result
    return value


def classify_health(health):
    """Identify known old service; do not infer undocumented encoder APIs."""
    if not isinstance(health, dict): raise ValueError("Health response must be an object")
    _finite_json(health)
    ok = health.get("status") == "ok" or (health.get("status") is None and health.get("message") == "pong")
    models = health.get("models", [])
    if not isinstance(models, list): models = []
    advertised = [x for x in models if isinstance(x, str)]
    model = health.get("model")
    if isinstance(model, str) and model not in advertised: advertised.append(model)
    claimed_encoder = any(key in health for key in (
        "encoder", "encoder_model", "encoder_model_id", "encoder_loaded",
        "raw_video", "raw_video_supported", "supports_raw_video", "video_encoder",
    ))
    capabilities = health.get("capabilities")
    if isinstance(capabilities, dict):
        claimed_encoder = claimed_encoder or any("video" in key.lower() or "encoder" in key.lower() for key in capabilities)
    legacy = (health.get("features_expected") == 250 and
              "v2_tabular_xgboost" in advertised and not claimed_encoder)
    if not ok:
        return {"status": "server_error", "encoder_verified": False,
                "reason": "The service replied but did not return a healthy pong.",
                "advertised_models": advertised}
    if claimed_encoder:
        return {"status": "encoder_protocol_unverified", "encoder_verified": False,
                "reason": "Health advertises encoder-related metadata. Its raw-frame API, preprocessing and real inference still require verification.",
                "advertised_models": advertised, "service_kind": "encoder_claim_unverified"}
    return {"status": "needs_encoder_protocol", "encoder_verified": False,
            "service_kind": "legacy_head_and_tabular" if legacy else "unknown_pong_service",
            "fusion_head_loaded": bool(health.get("jepa_loaded")) if legacy else None,
            "reason": ("This is the reviewed feature-only fusion/tabular socket. jepa_loaded describes its small fusion head, not a raw-video V-JEPA encoder."
                       if legacy else "The pong does not advertise a verifiable raw-video encoder contract."),
            "advertised_models": advertised}


class SocketHealthClient:
    def __init__(self, host, port=5002, timeout_seconds=3.0, max_response_bytes=65536):
        self.address = ipaddress.ip_address(host)
        if self.address.is_unspecified or self.address.is_multicast or str(self.address) == "255.255.255.255":
            raise ValueError("Use a specific resolved peer IP, not a bind or broadcast address")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("Port must be within 1..65535")
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30:
            raise ValueError("Total timeout must be within 0..30 seconds")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or not 128 <= max_response_bytes <= 1048576:
            raise ValueError("Health response limit must be 128..1048576 bytes")
        self.host, self.port = str(self.address), port
        self.timeout = float(timeout_seconds)
        self.max_response = max_response_bytes

    def probe(self):
        started = time.monotonic()
        deadline = started + self.timeout
        result = {"host": self.host, "port": self.port, "reachable": False,
                  "request_kind": "metadata_only_ping", "participant_data_sent": False,
                  "encoder_verified": False}
        def remaining():
            left = deadline - time.monotonic()
            if left <= 0: raise TimeoutError("Total health deadline exceeded")
            return left
        connected = False
        try:
            family = socket.AF_INET6 if self.address.version == 6 else socket.AF_INET
            with socket.socket(family, socket.SOCK_STREAM) as stream:
                stream.settimeout(remaining())
                stream.connect((self.host, self.port))
                connected = True
                stream.settimeout(remaining())
                stream.sendall(PING)
                buffer = bytearray()
                while b"\n" not in buffer:
                    stream.settimeout(remaining())
                    chunk = stream.recv(min(16384, self.max_response + 1 - len(buffer)))
                    if not chunk: break
                    buffer.extend(chunk)
                    if len(buffer) > self.max_response:
                        raise ValueError("Health response exceeds configured byte limit")
                if not buffer: raise ValueError("Empty health response")
                line = bytes(buffer).split(b"\n", 1)[0]
                health = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs,
                                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON constant")))
                result.update(classify_health(health), reachable=True,
                              health=_safe_metadata(health), response_bytes=len(buffer))
        except (TimeoutError, socket.timeout):
            result.update(status="timeout", reachable=connected, reason="Metadata-only health probe exceeded its total deadline or socket timeout.")
        except (UnicodeError, ValueError, RecursionError) as exc:
            result.update(status="protocol_error", reachable=connected,
                          reason="Invalid bounded health response: " + str(exc)[:180])
        except OSError as exc:
            result.update(status="network_error", reachable=connected,
                          reason=type(exc).__name__ + (" errno=" + str(exc.errno) if exc.errno is not None else ""))
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        return result


def probe_socket_health(host, port=5002, timeout_seconds=3.0, max_response_bytes=65536):
    return SocketHealthClient(host, port, timeout_seconds, max_response_bytes).probe()


# This is a NEW, explicitly versioned contract. The committed friend server does
# not implement it. No compatibility fallback to its random-input jepa path exists.
VISION_PROTOCOL = "sunhacks-video-encoder-v1"
VISION_POOLING = "spatial_mean_per_tubelet"
VISION_MODEL_SPECS = {
    "facebook/vjepa2-vitl-fpc64-256": {
        "spatial_size": [256, 256],
        "preprocessing_id": "vjepa2-vitl-fpc64-256-official-v1",
    },
    "facebookresearch/vjepa2:vjepa2_1_vit_large_384": {
        "spatial_size": [384, 384],
        "preprocessing_id": "vjepa2_1-vit-large-384-official-v1",
    },
}
MAX_VIDEO_REQUEST_BYTES = 16 * 1024 * 1024
MAX_VIDEO_RESPONSE_BYTES = 1024 * 1024


class RemoteVisionError(RuntimeError):
    """Controlled remote-video failure; never a reason to fabricate features."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _positive_number(value, name, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value <= 0 or
            (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be finite and positive" +
                         (f" and at most {maximum}" if maximum is not None else ""))
    return float(value)


def _revision(value):
    return isinstance(value, str) and 0 < len(value) <= 200 and not any(ord(c) < 32 for c in value)


def _sha256(value):
    import re
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _cuda_device(value):
    import re
    return isinstance(value, str) and re.fullmatch(r"cuda(?::[0-9]+)?", value) is not None


def _validate_preprocessing(value, model_id, expected_sha256=None):
    spec = VISION_MODEL_SPECS[model_id]
    if not isinstance(value, dict):
        raise RemoteVisionError("protocol_error", "Missing preprocessing identity")
    expected = {
        "id": spec["preprocessing_id"],
        "input_stage": "transport_rgb_uint8",
        "applied_once": True,
        "spatial_size": spec["spatial_size"],
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise RemoteVisionError("protocol_error", "Preprocessing stage, size or identity does not match the selected base model")
    # Equality alone would accept 1 for True; boolean provenance must be explicit.
    if value.get("applied_once") is not True or not _sha256(value.get("config_sha256")):
        raise RemoteVisionError("protocol_error", "Preprocessing requires an explicit once-only flag and a configuration SHA256")
    if expected_sha256 is not None and value["config_sha256"] != expected_sha256:
        raise RemoteVisionError("protocol_error", "Preprocessing configuration SHA256 differs from the reviewed pin")
    return {"id": value["id"], "input_stage": value["input_stage"],
            "applied_once": True, "spatial_size": list(value["spatial_size"]),
            "config_sha256": value["config_sha256"]}


class RemoteVisionClient:
    """Bounded raw-video client, disabled until a matching server is reviewed.

    Call probe() first. Its metadata can establish a declared contract, never
    independently prove encoder execution. Then a source reviewer must pin the
    model revision and preprocessing SHA256 and set protocol_reviewed=True.
    encode() sends real RGB frames only when all those gates pass. No socket is
    opened on construction or import, and there is no legacy fallback/retry.
    """
    def __init__(self, host, port=5002, *, expected_model_id,
                 expected_revision=None, expected_preprocessing_sha256=None,
                 protocol_reviewed=False, timeout_seconds=8,
                 shared_token_env=None):
        self.timeout = _positive_number(timeout_seconds, "Total video deadline", 30)
        self.health_client = SocketHealthClient(host, port, timeout_seconds=min(self.timeout, 3))
        if not isinstance(expected_model_id, str) or expected_model_id not in VISION_MODEL_SPECS:
            raise ValueError("Select an explicit supported V-JEPA2 or V-JEPA2.1 identity")
        if not isinstance(protocol_reviewed, bool):
            raise ValueError("protocol_reviewed must be a boolean")
        if expected_revision is not None and not _revision(expected_revision):
            raise ValueError("Expected revision must be a bounded, nonempty revision or hash")
        if expected_preprocessing_sha256 is not None and not _sha256(expected_preprocessing_sha256):
            raise ValueError("Expected preprocessing SHA256 must be 64 lowercase hex characters")
        if shared_token_env not in (None, "SUNHACKS_SOCKET_TOKEN"):
            raise ValueError("Only the separate SUNHACKS_SOCKET_TOKEN environment variable is supported")
        self.host = self.health_client.host
        self.port = self.health_client.port
        self.expected_model_id = expected_model_id
        self.expected_revision = expected_revision
        self.expected_preprocessing_sha256 = expected_preprocessing_sha256
        self.protocol_reviewed = protocol_reviewed
        self.shared_token_env = shared_token_env
        self._declared_encoder = None
        self._declaration_fingerprint = None

    def _fingerprint(self):
        return (self.host, self.port, self.health_client.host, self.health_client.port,
                str(self.health_client.address), self.expected_model_id, self.expected_revision,
                self.expected_preprocessing_sha256, self.shared_token_env,
                json.dumps(self._declared_encoder, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def _endpoint_matches_health(self):
        return (self.host == self.health_client.host == str(self.health_client.address) and
                type(self.port) is int and 1 <= self.port <= 65535 and
                type(self.health_client.port) is int and self.port == self.health_client.port and
                not self.health_client.address.is_unspecified and
                not self.health_client.address.is_multicast and self.host != "255.255.255.255")

    def probe(self):
        """Metadata-only discovery. A positive declaration remains unverified."""
        if not self._endpoint_matches_health():
            self._declared_encoder = None
            self._declaration_fingerprint = None
            return {"status": "configuration_error", "reachable": False,
                    "encoder_verified": False, "participant_data_sent": False,
                    "reason": "Health and video endpoints differ; construct a new client for the selected peer"}
        result = self.health_client.probe()
        self._declared_encoder = None
        self._declaration_fingerprint = None
        health = result.get("health")
        if not isinstance(health, dict) or result.get("status") in ("network_error", "timeout", "protocol_error", "server_error"):
            return result
        try:
            protocols = health.get("protocols")
            if not isinstance(protocols, list) or VISION_PROTOCOL not in protocols:
                raise RemoteVisionError("needs_encoder_protocol", "Peer does not advertise the explicit video encoder protocol")
            if health.get("auth_required") is True and self.shared_token_env is None:
                raise RemoteVisionError("auth_configuration", "Peer requires its separate socket token before video can be sent")
            encoder = health.get("encoder")
            if not isinstance(encoder, dict):
                raise RemoteVisionError("needs_encoder_protocol", "Peer lacks the required encoder capability object")
            if (encoder.get("loaded") is not True or
                    encoder.get("real_encoder") is not True or
                    encoder.get("synthetic") is not False or
                    encoder.get("model_id") != self.expected_model_id or
                    not _revision(encoder.get("revision")) or
                    not _cuda_device(encoder.get("device")) or
                    type(encoder.get("input_frames")) is not int or encoder.get("input_frames") != 64 or
                    encoder.get("output_sequence_shape") != [32, 1024] or
                    any(type(item) is not int for item in encoder.get("output_sequence_shape", [])) or
                    encoder.get("pooling") != VISION_POOLING):
                raise RemoteVisionError("needs_encoder_protocol", "Peer encoder declaration is incomplete or differs from the selected model/shape/pooling")
            if self.expected_revision is not None and encoder["revision"] != self.expected_revision:
                raise RemoteVisionError("needs_encoder_protocol", "Peer model revision differs from the reviewed pin")
            _validate_preprocessing(encoder.get("preprocessing"), self.expected_model_id, self.expected_preprocessing_sha256)
            self._declared_encoder = encoder
            self._declaration_fingerprint = self._fingerprint()
            result.update(status="encoder_protocol_unverified", encoder_verified=False,
                          video_protocol=VISION_PROTOCOL, encoder_contract_declared=True,
                          reason="Peer declares the requested video contract. Source review and a genuine encoder fixture are still required before participant use.")
        except RemoteVisionError as exc:
            result.update(status=exc.code, encoder_verified=False,
                          encoder_contract_declared=False, reason=str(exc))
        return result

    def _check_ready(self):
        if not self._endpoint_matches_health():
            self._declared_encoder = None
            self._declaration_fingerprint = None
            raise RemoteVisionError("needs_encoder_protocol", "Endpoint configuration changed; construct a new client before video")
        if not self.protocol_reviewed:
            raise RemoteVisionError("review_required", "The new GPU encoder protocol/source has not been reviewed; video sending is disabled")
        if self.expected_revision is None or self.expected_preprocessing_sha256 is None:
            raise RemoteVisionError("review_required", "Pin the reviewed model revision and preprocessing SHA256 before enabling video")
        if self._declared_encoder is None:
            raise RemoteVisionError("needs_encoder_protocol", "A successful metadata-only probe must declare the selected video contract before encoding")
        if self._declaration_fingerprint != self._fingerprint():
            self._declared_encoder = None
            self._declaration_fingerprint = None
            raise RemoteVisionError("needs_encoder_protocol", "Endpoint, artifact pins or capabilities changed; construct/reprobe a reviewed client before video")

    def encode(self, clip, *, window_id, timestamps=None, sample_fps):
        """Return one validated vision branch, with private NumPy embeddings.

        clip: RGB uint8[64,H,W,3], never normalized/cropped model tensors.
        timestamps: strictly increasing client monotonic seconds, or omitted for
        a labeled relative-clip timeline. Exactly one request/response is made.
        """
        self._check_ready()
        import base64
        import io
        import os
        import uuid
        import numpy as np
        from PIL import Image

        started = time.monotonic()
        deadline = started + self.timeout
        def remaining():
            left = deadline - time.monotonic()
            if left <= 0:
                raise RemoteVisionError("timeout", "Total video encode/send/receive deadline exceeded")
            return left

        if (not isinstance(clip, np.ndarray) or clip.dtype != np.uint8 or
                clip.ndim != 4 or clip.shape[0] != 64 or clip.shape[-1] != 3 or
                min(clip.shape[1:3]) < 1):
            raise RemoteVisionError("input_error", "Requires exactly 64 RGB uint8 frames shaped [64,H,W,3]")
        if clip.nbytes > 512 * 1024 * 1024:
            raise RemoteVisionError("input_error", "Input clip exceeds the bounded local packing size")
        if type(window_id) is not int or not 0 <= window_id <= 2**53 - 1:
            raise RemoteVisionError("input_error", "window_id must be a nonnegative JSON-safe integer")
        try:
            fps = _positive_number(sample_fps, "sample_fps", 240)
        except ValueError as exc:
            raise RemoteVisionError("input_error", str(exc)) from None
        if timestamps is None:
            stamp_values = [index / fps for index in range(64)]
            timestamp_basis = "relative_clip_seconds"
        else:
            if (not isinstance(timestamps, (list, tuple, np.ndarray)) or
                    (isinstance(timestamps, np.ndarray) and timestamps.ndim != 1) or len(timestamps) != 64):
                raise RemoteVisionError("input_error", "Exactly 64 frame timestamps are required")
            stamp_values = []
            for stamp in timestamps:
                if (isinstance(stamp, (bool, np.bool_)) or not isinstance(stamp, (int, float, np.integer, np.floating)) or
                        not np.isfinite(stamp) or not math.isfinite(float(stamp)) or stamp < 0):
                    raise RemoteVisionError("input_error", "Frame timestamps must be finite nonnegative seconds")
                stamp_values.append(float(stamp))
            if any(right <= left for left, right in zip(stamp_values, stamp_values[1:])):
                raise RemoteVisionError("input_error", "Frame timestamps must increase strictly; missing/duplicate frames are not padded")
            timestamp_basis = "client_monotonic_seconds"
        request_id = str(uuid.uuid4())
        height, width = clip.shape[1:3]
        scale = min(1.0, 640.0 / max(height, width))
        target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        frames = []
        # Bound incremental base64 size as well as the final serialized request.
        encoded_bytes = 0
        for frame, stamp in zip(clip, stamp_values):
            remaining()
            image = Image.fromarray(frame, mode="RGB")
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            stream = io.BytesIO()
            image.save(stream, format="JPEG", quality=90, subsampling=0, optimize=False)
            packed = base64.b64encode(stream.getvalue()).decode("ascii")
            encoded_bytes += len(packed)
            if encoded_bytes > MAX_VIDEO_REQUEST_BYTES:
                raise RemoteVisionError("request_too_large", "Packed video exceeds the 16 MiB request limit")
            frames.append({"timestamp": stamp, "jpeg_base64": packed})
        request = {
            "action": "encode_video", "protocol": VISION_PROTOCOL,
            "request_id": request_id, "window_id": window_id,
            "model_id": self.expected_model_id, "revision": self.expected_revision,
            "sample_fps": fps, "timestamp_basis": timestamp_basis,
            "max_server_ms": max(1, int(remaining() * 1000)),
            "input_stage": "transport_rgb_uint8", "frame_count": 64,
            "transport": {"format": "jpeg", "color": "RGB", "max_edge": 640,
                          "quality": 90, "subsampling": 0, "size": list(target_size)},
            "frames": frames,
        }
        if self.shared_token_env is not None:
            token = os.environ.get(self.shared_token_env)
            if not isinstance(token, str) or not token or len(token) > 4096:
                raise RemoteVisionError("auth_configuration", "The separate socket token is configured but unavailable or overlong")
            request["auth_token"] = token
        packed_request = json.dumps(request, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        if len(packed_request) > MAX_VIDEO_REQUEST_BYTES:
            raise RemoteVisionError("request_too_large", "Serialized video exceeds the 16 MiB request limit")
        packing_ms = (time.monotonic() - started) * 1000
        remaining()
        try:
            family = socket.AF_INET6 if self.health_client.address.version == 6 else socket.AF_INET
            with socket.socket(family, socket.SOCK_STREAM) as stream:
                stream.settimeout(remaining())
                stream.connect((self.host, self.port))
                stream.settimeout(remaining())
                stream.sendall(packed_request)
                buffer = bytearray()
                while b"\n" not in buffer:
                    stream.settimeout(remaining())
                    chunk = stream.recv(min(16384, MAX_VIDEO_RESPONSE_BYTES + 1 - len(buffer)))
                    if not chunk:
                        raise RemoteVisionError("protocol_error", "Video response closed before mandatory NDJSON delimiter")
                    buffer.extend(chunk)
                    if len(buffer) > MAX_VIDEO_RESPONSE_BYTES:
                        raise RemoteVisionError("protocol_error", "Video response exceeds the 1 MiB limit")
                line, _, tail = bytes(buffer).partition(b"\n")
                if tail.strip():
                    raise RemoteVisionError("protocol_error", "One video request must return exactly one NDJSON object")
                response = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs,
                                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON constant")))
                branch = self._validate_response(response, request_id, window_id, stamp_values, fps, timestamp_basis)
                remaining()
        except RemoteVisionError:
            raise
        except (TimeoutError, socket.timeout):
            raise RemoteVisionError("timeout", "Remote video socket exceeded the remaining total deadline") from None
        except OSError as exc:
            raise RemoteVisionError("network_error", type(exc).__name__ + (" errno=" + str(exc.errno) if exc.errno is not None else "")) from None
        except (UnicodeError, ValueError, OverflowError, RecursionError):
            raise RemoteVisionError("protocol_error", "Invalid bounded video response") from None
        branch.update(elapsed_ms=round((time.monotonic() - started) * 1000, 3),
                      client_packing_ms=round(packing_ms, 3), request_bytes=len(packed_request),
                      response_bytes=len(buffer))
        return branch

    def _validate_response(self, response, request_id, window_id, stamps, fps, timestamp_basis):
        import numpy as np
        if not isinstance(response, dict):
            raise RemoteVisionError("protocol_error", "Video response must be a JSON object")
        _finite_json(response)
        if response.get("status") != "ok":
            raise RemoteVisionError("server_error", "GPU service did not return a successful video encoder result")
        expected = {
            "protocol": VISION_PROTOCOL, "action": "encode_video", "request_id": request_id,
            "window_id": window_id, "model_id": self.expected_model_id,
            "revision": self.expected_revision, "real_encoder": True,
            "synthetic": False, "sequence_shape": [32, 1024],
            "pooling": VISION_POOLING, "timestamp_basis": timestamp_basis,
            "sample_fps": fps, "frame_timestamps": stamps,
        }
        if any(response.get(key) != value for key, value in expected.items()):
            raise RemoteVisionError("protocol_error", "Encoder response identity, source window, shape or pooling differs from the request")
        if (response.get("real_encoder") is not True or response.get("synthetic") is not False or
                type(response.get("window_id")) is not int or not _cuda_device(response.get("device"))):
            raise RemoteVisionError("protocol_error", "A real CUDA encoder result with explicit provenance is required")
        if (isinstance(response["sample_fps"], bool) or not isinstance(response["sample_fps"], (int, float)) or
                any(type(item) is not int for item in response["sequence_shape"]) or
                not isinstance(response["frame_timestamps"], list) or
                any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in response["frame_timestamps"])):
            raise RemoteVisionError("protocol_error", "Numeric shape, rate and timestamp echoes cannot contain booleans or nonnumeric values")
        preprocessing = _validate_preprocessing(response.get("preprocessing"), self.expected_model_id, self.expected_preprocessing_sha256)
        raw_sequence = response.get("sequence")
        if (not isinstance(raw_sequence, list) or len(raw_sequence) != 32 or
                any(not isinstance(row, list) or len(row) != 1024 for row in raw_sequence) or
                any(isinstance(value, bool) or not isinstance(value, (int, float))
                    for row in raw_sequence for value in row)):
            raise RemoteVisionError("protocol_error", "Encoder sequence must contain exactly 32 by 1024 numeric values")
        sequence = np.asarray(raw_sequence, dtype=np.float64)
        if not np.isfinite(sequence).all() or np.any(np.abs(sequence) > np.finfo(np.float32).max):
            raise RemoteVisionError("protocol_error", "Encoder features must remain finite in float32")
        sequence = np.ascontiguousarray(sequence, dtype=np.float32)
        timings = response.get("timings")
        if not isinstance(timings, dict):
            raise RemoteVisionError("protocol_error", "GPU stage timings are missing")
        validated_timings = {}
        for name in ("decode_ms", "preprocess_ms", "encoder_ms", "pooling_ms", "total_ms"):
            value = timings.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RemoteVisionError("protocol_error", "GPU stage timings must be finite nonnegative milliseconds")
            validated_timings[name] = float(value)
        if timings["encoder_ms"] <= 0 or timings["total_ms"] < timings["encoder_ms"]:
            raise RemoteVisionError("protocol_error", "GPU encoder timing must be positive and contained in total server time")
        return {
            "status": "base", "model_id": self.expected_model_id, "revision": response["revision"],
            "device": response["device"], "input_contract": VISION_PROTOCOL,
            "embedding_shape": [1024], "pooling": "mean_all_tokens",
            "sequence_pooling": VISION_POOLING, "preprocessing": preprocessing,
            "_embedding": sequence.mean(axis=0, dtype=np.float64).astype(np.float32),
            "_sequence_embedding": sequence, "remote_timings": validated_timings,
            "window_id": window_id, "request_id": request_id,
            "real_encoder": True, "synthetic": False, "remote_protocol_validated": True,
            "reason": "Remote base encoder response passed the reviewed protocol. Task classifier accuracy is unvalidated.",
        }


def make_vision_runner(client, *, sample_fps, timestamps_provider=None):
    """Adapt to vision_runner(clip, window_id), without enabling/probing a peer.

    Optional timestamps_provider(window_id) supplies the captured monotonic
    timestamps. Otherwise the request explicitly uses relative clip seconds.
    Offline/unreviewed service errors remain unavailable; no local fallback runs.
    """
    if not isinstance(client, RemoteVisionClient):
        raise ValueError("A configured RemoteVisionClient is required")
    _positive_number(sample_fps, "sample_fps", 240)
    if timestamps_provider is not None and not callable(timestamps_provider):
        raise ValueError("timestamps_provider must be callable or None")
    def run(clip, window_id):
        started = time.monotonic()
        try:
            timestamps = timestamps_provider(window_id) if timestamps_provider is not None else None
            return client.encode(clip, window_id=window_id, timestamps=timestamps, sample_fps=sample_fps)
        except RemoteVisionError as exc:
            return {"status": "unavailable", "model_id": client.expected_model_id,
                    "revision": client.expected_revision, "input_contract": VISION_PROTOCOL,
                    "embedding_shape": None, "window_id": window_id,
                    "remote_error_code": exc.code, "reason": str(exc),
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 3)}
        except Exception:
            return {"status": "error", "model_id": client.expected_model_id,
                    "revision": client.expected_revision, "input_contract": VISION_PROTOCOL,
                    "embedding_shape": None, "window_id": window_id,
                    "reason": "Remote vision adapter failed before a valid encoder result was available.",
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 3)}
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe metadata only; never send video, audio or synthetic JEPA inputs")
    parser.add_argument("--host", required=True, help="Resolved peer IPv4/IPv6 address")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--timeout", type=float, default=3)
    args = parser.parse_args()
    print(json.dumps(probe_socket_health(args.host, args.port, args.timeout), indent=2, allow_nan=False))
