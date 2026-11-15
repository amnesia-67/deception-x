"""
DeceptionX V2 Model (Branch 2 Tabular LOSO XGBoost) Socket Service
Accredited to Gemini.pdf Specifications.

Provides a standalone TCP socket server running on localhost / 0.0.0.0 allowing
external nodes to stream behavioral feature vectors into the v2 model, process
them with GPU acceleration on the NVIDIA RTX 5070, and receive classification
probabilities, log-odds margins, and TreeSHAP forensic explanations back over
the same socket connection.
"""

import os
import sys
import json
import time
import socket
import select
import struct
import threading
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List

# Gracefully import model dependencies
from src.tabular_branch import TabularLOSOXGBoost
from src.explainer import ForensicTreeSHAPExplainer
from src.calibration import BaselineCalibrator
from src.feature_schema import FEATURE_NAMES, DOMAIN_SLICES, get_feature_domain

# Client-supplied identifiers echoed back verbatim so a reply can be matched to the exact capture that
# produced it (a reply is otherwise bound only to its TCP connection).
ECHO_KEYS = ("request_id", "source_id", "session_id", "client_revision", "capture_started_at")


def echo_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in ECHO_KEYS:
        if k in payload:
            v = payload[k]
            if isinstance(v, bool) or not isinstance(v, (str, int, float)) or len(str(v)) > 128:
                raise ValueError(f"'{k}' must be a string or number of at most 128 characters")
            out[k] = v
    return out


SERVED_ARTIFACTS = ("models/ensemble_deploy.json", "models/branch2_xgboost.json", "models/branch2_digest_schema.json",
                    "models/vjepa2_1_l_lora/deploy_meta.json", "models/vjepa2_1-vitl-384/manifest.json",
                    "models/pilot_webcam/dasheng/deploy_meta.json", "models/pilot_webcam/vjepa2_1_l/deploy_meta.json",
                    "models/pilot_webcam/emotion2vec/deploy_meta.json", "models/pilot_webcam/fusion.json")


def server_revision() -> Dict[str, Any]:
    """What this server is running, taken once at startup (the code it imported and the artifacts it loaded).

    code               commit, content-based dirty flag, tracked-source tree hash, untracked source files
                       (src.provenance.git_revision: the same answer from Windows git or git inside WSL)
    artifacts_sha256   SHA256 of each served manifest/model file with CRLF -> LF, so a Windows checkout, WSL and an
                       LF clone (Mac) all compute the same value; for these JSON files it equals SHA256 of the git blob
    """
    import time

    from src.provenance import git_revision, sha256_lf_file

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = {rel: sha256_lf_file(rel) for rel in SERVED_ARTIFACTS if os.path.exists(os.path.join(root, rel))}
    return {"code": git_revision(), "artifacts_sha256": files,
            "artifacts_hash": "sha256 of file bytes with CRLF->LF (equals the git blob for these files)",
            "recorded_at_unix": time.time()}

try:
    import torch
    from src.multimodal_fusion import Branch1MultimodalNetwork
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Branch1MultimodalNetwork = None
    TORCH_AVAILABLE = False

DEFAULT_HOST = os.environ.get("V2_SOCKET_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("V2_SOCKET_PORT", "5002"))
DEFAULT_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "branch2_digest_schema.json")
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "branch2_xgboost.json"
)


class V2ModelServer:
    """
    High-performance TCP Socket Server serving the DeceptionX V2 model (Branch 2 XGBoost).
    Supports single samples, batches, newline-delimited JSON (NDJSON), raw JSON,
    and binary packed float arrays.
    """
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cuda",
        enable_jepa: bool = True,
        enable_webcam_pilot: bool = False
    ):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.requested_device = device
        self.active_device = device
        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.listen_thread: Optional[threading.Thread] = None

        # Feature schema of the deployed (real-data) digest model; falls back to the PDF's 250-dim layout.
        self.schema = json.load(open(DEFAULT_SCHEMA_PATH)) if os.path.exists(DEFAULT_SCHEMA_PATH) else None
        self.feature_names = list(self.schema["feature_names"]) if self.schema else list(FEATURE_NAMES)
        self.n_features = len(self.feature_names)
        # Load model & explainer
        self.model: Optional[TabularLOSOXGBoost] = None
        self.explainer: Optional[ForensicTreeSHAPExplainer] = None
        self.calibrator = BaselineCalibrator()
        self._load_model()

        # Load Branch 1 (V-JEPA 2 Neural Discrepancy Network)
        self.branch1_model: Optional[Any] = None
        self.branch1_path = os.path.join(
            os.path.dirname(self.model_path),
            "branch1_fusion.pt"
        )
        self._load_branch1_model()

        # Best model: stacked ensemble (tuned dasheng + behavioural digest) on frames + audio + transcript
        self.ensemble = None
        self.revision = server_revision()
        if TORCH_AVAILABLE:
            try:
                from src.ensemble_service import EnsembleService
                self.ensemble = EnsembleService(device="cuda" if torch.cuda.is_available() else "cpu")
                print(f"[V2 Socket Server] Stacked ensemble loaded: {self.ensemble.meta['branches']} "
                      f"(nested CV AUROC {self.ensemble.meta['cv_metrics']['auroc']:.3f})")
            except Exception as e:
                print(f"[V2 Socket Server] Stacked ensemble not loaded: {e}")

        # Live V-JEPA 2.1 ViT-L (LoRA, 5-fold classifier ensemble) on raw video frames; no ViT-B / V-JEPA 2 fallback
        self.jepa = None
        self.jepa_unavailable_reason = "disabled" if not enable_jepa else "not loaded"
        if enable_jepa and TORCH_AVAILABLE:
            try:
                from src.jepa_service import JEPAService
                self.jepa = JEPAService(device="cuda" if torch.cuda.is_available() else "cpu")
                print(f"[V2 Socket Server] V-JEPA 2.1 ViT-L video model loaded: {self.jepa.info()['folds']} fold models "
                      f"(CV AUROC {(self.jepa.info()['cv_metrics'] or {}).get('auroc')})")
            except Exception as e:
                self.jepa_unavailable_reason = str(e)
                print(f"[V2 Socket Server] V-JEPA 2.1 ViT-L video model unavailable (jepa_video disabled, no fallback): {e}")

        # Opt-in webcam-pilot ViT-L (models/pilot_webcam/vjepa2_1_l). Only loaded with --webcam-pilot and only used
        # when a jepa_video request names "profile": "webcam_pilot"; the served default never changes.
        self.pilot_video = None
        self.pilot_unavailable_reason = "not enabled (start with --webcam-pilot)"
        if enable_webcam_pilot and TORCH_AVAILABLE:
            try:
                from src.pilot_webcam import PilotVideoService
                self.pilot_video = PilotVideoService(device="cuda" if torch.cuda.is_available() else "cpu")
                print(f"[V2 Socket Server] webcam-pilot ViT-L loaded: {self.pilot_video.info()['model']}")
            except Exception as e:
                self.pilot_unavailable_reason = str(e)
                print(f"[V2 Socket Server] webcam-pilot ViT-L unavailable: {e}")

    def _load_branch1_model(self):
        """Loads Branch 1 V-JEPA neural discrepancy network if PyTorch and checkpoint are available."""
        if not TORCH_AVAILABLE:
            print("[V2 Socket Server] PyTorch not installed in this environment; running in tabular-only mode.")
            return

        if not os.path.exists(self.branch1_path):
            print(f"[V2 Socket Server] Notice: Branch 1 checkpoint not found at {self.branch1_path}")
            return

        print(f"[V2 Socket Server] Loading Branch 1 JEPA Neural Model from {self.branch1_path}...")
        torch_dev = "cuda" if (self.active_device == "cuda" and torch.cuda.is_available()) else "cpu"
        try:
            net = Branch1MultimodalNetwork(
                visual_dim=1024,
                audio_dim=768,
                text_dim=128,
                d=256
            ).to(torch_dev)
            ckpt = torch.load(self.branch1_path, map_location=torch_dev)
            if isinstance(ckpt, dict) and "network_state_dict" in ckpt:
                net.load_state_dict(ckpt["network_state_dict"])
            elif isinstance(ckpt, dict):
                net.load_state_dict(ckpt)
            net.eval()
            self.branch1_model = net
            print(f"[V2 Socket Server] Branch 1 JEPA neural model loaded successfully on device: {torch_dev}")
        except Exception as e:
            print(f"[V2 Socket Server] Warning: Failed to load Branch 1 JEPA model: {e}")

    def _load_model(self):
        """Loads Branch 2 model and initializes GPU/CPU runtime and TreeSHAP explainer."""
        print(f"[V2 Socket Server] Loading V2 model from {self.model_path}...")
        
        # Try requested device first (e.g. cuda), fall back to cpu if unavailable
        loaded = False
        devices_to_try = [self.requested_device]
        if self.requested_device != "cpu":
            devices_to_try.append("cpu")

        for dev in devices_to_try:
            try:
                m = TabularLOSOXGBoost(device=dev)
                if os.path.exists(self.model_path):
                    m.load_model(self.model_path)
                    self.model = m
                    self.active_device = dev
                    self.explainer = ForensicTreeSHAPExplainer(
                        self.model, self.feature_names if self.schema else None)
                    print(f"[V2 Socket Server] V2 model loaded successfully on device: {self.active_device}")
                    loaded = True
                    break
                else:
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
            except Exception as e:
                print(f"[V2 Socket Server] Warning: Failed to load on device {dev}: {e}")

        if not loaded:
            raise RuntimeError(f"Could not load V2 model from {self.model_path} on any device.")

    def parse_payload(self, data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Parses incoming socket byte buffer into an (N, n_features) numpy array and metadata
        (n_features comes from models/branch2_digest_schema.json for the real-data model).
        Accepts:
        - JSON string: {"z": [...]}, {"features": [...]}, [f1, f2, ...], or {"batch": [[...]]}
        - JSON named features: {"features": {"<name>": value, ...}}; {"action": "schema"} lists names
        - Binary payload: n_features * float32 or n_features * float64 bytes
        - Comma/space-delimited numbers
        """
        raw_text = None
        # 1. Try binary float payload
        n = self.n_features
        if len(data) == 4 * n:
            arr = np.frombuffer(data, dtype=np.float32).reshape(1, n)
            return arr, {"format": "binary_float32"}
        elif len(data) == 8 * n:
            arr = np.frombuffer(data, dtype=np.float64).astype(np.float32).reshape(1, n)
            return arr, {"format": "binary_float64"}

        # 2. Try text / JSON
        try:
            raw_text = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            raise ValueError(f"Payload not valid UTF-8 and does not match binary float shapes ({len(data)} bytes)")

        if not raw_text:
            raise ValueError("Empty request payload")

        # 3. Parse JSON
        if raw_text.startswith("{") or raw_text.startswith("["):
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as err:
                raise ValueError(f"Malformed JSON payload: {err}")

            # Check if ping/health action
            if isinstance(parsed, dict) and (parsed.get("action") == "ping" or parsed.get("ping") is True):
                return None, {"action": "ping"}
            if isinstance(parsed, dict) and parsed.get("action") == "schema":
                return None, {"action": "schema"}
            if isinstance(parsed, dict) and parsed.get("action") == "analyze":
                return None, {"action": "analyze", "payload": parsed}
            if isinstance(parsed, dict) and parsed.get("action") == "jepa_video":
                return None, {"action": "jepa_video", "payload": parsed}
            # Named features: {"features": {"txt_hedge_per100": 1.2, ...}} (missing names -> NaN)
            if isinstance(parsed, dict) and isinstance(parsed.get("features"), dict):
                named = parsed["features"]
                unknown = [k for k in named if k not in self.feature_names]
                if unknown:
                    raise ValueError(f"Unknown feature names: {unknown[:5]}")
                row = [float(named.get(k, np.nan)) for k in self.feature_names]
                return np.array([row], dtype=np.float32), {
                    "format": "json_named", "batch_size": 1,
                    "missing": [k for k in self.feature_names if k not in named]}

            # Check if JEPA / Branch 1 action
            if isinstance(parsed, dict) and (
                parsed.get("action") in ("jepa", "branch1")
                or parsed.get("model") in ("jepa", "branch1", "branch1_jepa_neural")
                or ("f_v" in parsed or "visual" in parsed)
            ):
                return None, {
                    "action": "jepa",
                    "f_v": parsed.get("f_v") or parsed.get("visual"),
                    "f_a": parsed.get("f_a") or parsed.get("audio"),
                    "f_t": parsed.get("f_t") or parsed.get("text")
                }

            features = None
            normalize = False

            if isinstance(parsed, list):
                features = parsed
            elif isinstance(parsed, dict):
                normalize = parsed.get("normalize", False)
                if "z" in parsed:
                    features = parsed["z"]
                elif "features" in parsed:
                    features = parsed["features"]
                elif "data" in parsed:
                    features = parsed["data"]
                elif "raw_features" in parsed:
                    features = parsed["raw_features"]
                    normalize = True
                elif "batch" in parsed:
                    features = parsed["batch"]
                else:
                    raise ValueError(f"JSON dict missing feature key ('features', 'z', 'data', 'batch'). Keys provided: {list(parsed.keys())}")

            arr = np.array(features, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            if normalize:
                if not self.calibrator.calibrated:
                    raise ValueError("raw_features need a real subject calibration first; "
                                     "no synthetic default baseline is used")
                norm_rows = [self.calibrator.normalize(row) for row in arr]
                arr = np.array(norm_rows, dtype=np.float32)

            if arr.shape[1] != self.n_features:
                raise ValueError(f"Expected {self.n_features} features, got {arr.shape[1]}")

            return arr, {"format": "json", "batch_size": int(arr.shape[0])}

        # 4. Try comma/space separated text
        parts = [p.strip() for p in raw_text.replace(",", " ").split() if p.strip()]
        if len(parts) == self.n_features:
            arr = np.array([float(x) for x in parts], dtype=np.float32).reshape(1, self.n_features)
            return arr, {"format": "plain_delimited", "batch_size": 1}

        raise ValueError(f"Unrecognized payload structure. Expected {self.n_features} features, parsed {len(parts)} tokens.")

    def process_features(self, X: np.ndarray, top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Runs features through V2 model (predict_proba) and ForensicTreeSHAPExplainer.
        """
        assert self.model is not None and self.model.is_fitted
        results = []
        t0 = time.perf_counter()

        probs = self.model.predict_proba(X)
        for i in range(len(X)):
            sample_z = X[i:i+1]
            p = float(probs[i])
            verdict = "Deceptive" if p >= 0.50 else "Truthful"
            confidence = p if p >= 0.50 else (1.0 - p)

            sample_res: Dict[str, Any] = {
                "sample_index": i,
                "model": "v2_tabular_xgboost",
                "branch": 2,
                "device": self.active_device,
                "probability": round(p, 6),
                "p_xgb": round(p, 6),
                "prediction": int(p >= 0.50),
                "verdict": verdict,
                "confidence": round(float(confidence), 6)
            }

            if self.explainer is not None:
                try:
                    shap = self.explainer.explain(sample_z, top_k=top_k)
                    sample_res["margin"] = round(float(shap["margin"]), 4)
                    sample_res["base_value_phi_0"] = round(float(shap["base_value_phi_0"]), 4)
                    sample_res["top_deceptive_drivers"] = shap["top_positive_drivers"]
                    sample_res["top_truthful_drivers"] = shap["top_negative_drivers"]
                    sample_res["domain_attributions"] = {
                        k: round(v, 4) for k, v in shap["domain_attributions"].items()
                    }
                except Exception as e:
                    sample_res["shap_error"] = str(e)

            results.append(sample_res)

        total_ms = (time.perf_counter() - t0) * 1000.0
        for r in results:
            r["batch_total_ms"] = round(total_ms, 3)
            r["per_sample_ms"] = round(total_ms / max(1, len(X)), 3)

        return results

    def process_jepa(
        self,
        f_v: Optional[Union[List, np.ndarray]] = None,
        f_a: Optional[Union[List, np.ndarray]] = None,
        f_t: Optional[Union[List, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Runs inference on Branch 1 (V-JEPA 2 Neural Discrepancy & Fusion Network).
        Takes:
          f_v: visual tubelets shape (B, 32, 1024) or (32, 1024)
          f_a: audio features shape (B, 32, 768) or (32, 768)
          f_t: text embedding shape (B, 128) or (128,)
        All three inputs are required.
        """
        if self.branch1_model is None:
            raise RuntimeError("Branch 1 JEPA neural model is not loaded in this environment.")

        dev = next(self.branch1_model.parameters()).device
        t0 = time.perf_counter()

        # Visual tensor (B, 32, 1024)
        if f_v is None:
            raise ValueError("missing input F_v: random placeholder inputs are not allowed")
        else:
            F_v = torch.tensor(f_v, dtype=torch.float32, device=dev)
            if F_v.ndim == 2:
                F_v = F_v.unsqueeze(0)

        # Audio tensor (B, 32, 768)
        if f_a is None:
            raise ValueError("missing input F_a: random placeholder inputs are not allowed")
        else:
            F_a = torch.tensor(f_a, dtype=torch.float32, device=dev)
            if F_a.ndim == 2:
                F_a = F_a.unsqueeze(0)

        # Text tensor (B, 128)
        if f_t is None:
            raise ValueError("missing input f_t_tensor: random placeholder inputs are not allowed")
        else:
            f_t_tensor = torch.tensor(f_t, dtype=torch.float32, device=dev)
            if f_t_tensor.ndim == 1:
                f_t_tensor = f_t_tensor.unsqueeze(0)

        with torch.no_grad():
            out = self.branch1_model(F_v, F_a, f_t_tensor)

        p_neural = float(out["p_neural"][0].cpu().item())
        alpha = out["alpha"][0].cpu().tolist()
        d_va = float(out["D_va"][0].cpu().item())
        d_vl = float(out["D_vl"][0].cpu().item())
        d_al = float(out["D_al"][0].cpu().item())

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        verdict = "Deceptive" if p_neural >= 0.50 else "Truthful"
        confidence = p_neural if p_neural >= 0.50 else (1.0 - p_neural)

        return {
            "model": "branch1_jepa_neural",
            "branch": 1,
            "device": str(dev),
            "probability": round(p_neural, 6),
            "p_neural": round(p_neural, 6),
            "prediction": int(p_neural >= 0.50),
            "verdict": verdict,
            "confidence": round(confidence, 6),
            "modality_reliability_weights": {
                "alpha_visual": round(alpha[0], 4),
                "alpha_audio": round(alpha[1], 4),
                "alpha_text": round(alpha[2], 4)
            },
            "cross_modal_discrepancy": {
                "d_va": round(d_va, 4),
                "d_vl": round(d_vl, 4),
                "d_al": round(d_al, 4)
            },
            "inference_ms": round(elapsed_ms, 3)
        }

    def _process_and_reply(self, client_sock: socket.socket, raw_payload: bytes):
        """Processes payload bytes and writes JSON response back to client socket."""
        echo = {}
        received = time.time()
        try:
            X, meta = self.parse_payload(raw_payload)
            if isinstance(meta.get("payload"), dict):
                echo = echo_fields(meta["payload"])
            if meta.get("action") == "ping":
                response = {
                    "status": "ok",
                    "message": "pong",
                    "server_revision": self.revision,
                    "model": "v2_tabular_xgboost",
                    "branch": 2,
                    "device": self.active_device,
                    "features_expected": self.n_features,
                    "trained_on": (self.schema or {}).get("trained_on", "unknown"),
                    "ensemble_loaded": self.ensemble is not None,
                    "ensemble": self.ensemble.info() if self.ensemble is not None else None,
                    "jepa_loaded": self.jepa is not None,
                    "jepa_unavailable_reason": None if self.jepa is not None else self.jepa_unavailable_reason,
                    "jepa": self.jepa.info() if self.jepa is not None else None,
                    "models": ["v2_tabular_xgboost"] + (["stacked_ensemble"] if self.ensemble is not None else [])
                              + ([self.jepa.info()["model"]] if self.jepa is not None else []),
                    "profiles": {"served": "default", "webcam_pilot": (self.pilot_video.info() if self.pilot_video
                                                                       is not None else self.pilot_unavailable_reason)},
                    "timestamp": time.time()
                }
            elif meta.get("action") == "schema":
                response = {"status": "ok", "feature_names": self.feature_names,
                            "cv_metrics": (self.schema or {}).get("cv_metrics"),
                            "caveats": (self.schema or {}).get("caveats")}
            elif meta.get("action") == "analyze":
                if self.ensemble is None:
                    raise RuntimeError("stacked ensemble is not loaded on this server")
                response = {"status": "success", **echo, **self.ensemble.predict(meta["payload"]),
                            "server_revision": self.revision, "received_at_unix": received}
            elif meta.get("action") == "jepa_video" and meta["payload"].get("profile", "served") != "served":
                if meta["payload"]["profile"] != "webcam_pilot":
                    raise ValueError("profile must be 'served' (default) or 'webcam_pilot'")
                if self.pilot_video is None:
                    raise RuntimeError(f"webcam_pilot profile unavailable: {self.pilot_unavailable_reason}")
                response = {"status": "success", **echo, **self.pilot_video.predict(meta["payload"]),
                            "server_revision": self.revision, "received_at_unix": received}
            elif meta.get("action") == "jepa_video":
                if self.jepa is None:
                    raise RuntimeError("V-JEPA 2.1 ViT-L video model is unavailable on this server (no ViT-B / V-JEPA 2 "
                                       f"fallback): {self.jepa_unavailable_reason}")
                response = {"status": "success", **echo, **self.jepa.predict(meta["payload"]),
                            "server_revision": self.revision, "received_at_unix": received}
            elif meta.get("action") == "jepa":
                res = self.process_jepa(
                    f_v=meta.get("f_v"),
                    f_a=meta.get("f_a"),
                    f_t=meta.get("f_t")
                )
                response = {
                    "status": "success",
                    **res
                }
            else:
                inferences = self.process_features(X)
                if len(inferences) == 1:
                    response = {
                        "status": "success",
                        **inferences[0]
                    }
                else:
                    response = {
                        "status": "success",
                        "batch_size": len(inferences),
                        "results": inferences
                    }
        except Exception as err:
            response = {
                "status": "error",
                "model": "v2_tabular_xgboost",
                "error": str(err),
                **echo,
            }

        resp_bytes = json.dumps(response).encode("utf-8") + b"\n"
        client_sock.sendall(resp_bytes)

    def handle_client(self, client_sock: socket.socket, client_addr: Tuple[str, int]):
        """
        Handles an individual client connection.
        Supports both persistent streaming sessions (NDJSON) and single query-response.
        """
        buffer = b""
        client_sock.settimeout(120.0)  # idle timeout (video requests take a few seconds)

        try:
            while self.running:
                chunk = client_sock.recv(65536)
                if not chunk:
                    # Connection closed by client
                    break
                buffer += chunk

                # Check if buffer has framed binary payload (4-byte length prefix)
                if len(buffer) >= 4:
                    prefix_len = struct.unpack(">I", buffer[:4])[0]
                    if prefix_len in (1000, 2000) and len(buffer) >= 4 + prefix_len:
                        payload = buffer[4:4 + prefix_len]
                        buffer = buffer[4 + prefix_len:]
                        self._process_and_reply(client_sock, payload)
                        continue

                # Check if buffer has complete lines / JSON payloads
                while b"\n" in buffer or (len(buffer) < 65536 and buffer.startswith(b"{")
                                           and buffer.rstrip().endswith(b"}")):
                    if b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                    else:
                        line = buffer.strip()
                        buffer = b""

                    self._process_and_reply(client_sock, line)

            # Process any remaining buffer on disconnect
            if buffer.strip():
                try:
                    X, meta = self.parse_payload(buffer.strip())
                    if meta.get("action") == "ping":
                        response = {"status": "ok", "message": "pong", "model": "v2_tabular_xgboost",
                                    "server_revision": self.revision}
                    else:
                        inferences = self.process_features(X)
                        response = {"status": "success", **(inferences[0] if len(inferences) == 1 else {"results": inferences})}
                except Exception as err:
                    response = {"status": "error", "error": str(err)}
                client_sock.sendall(json.dumps(response).encode("utf-8") + b"\n")

        except (socket.timeout, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                client_sock.close()
            except Exception:
                pass

    def start(self, block: bool = True):
        """Starts the socket server listening for connections."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(128)
        self.running = True

        actual_addr = self.server_socket.getsockname()
        print(f"============================================================")
        print(f"[DeceptionX Model Socket Server]")
        print(f"Listening on: {self.host}:{self.port} (bound {actual_addr})")
        print(f"Model V2: Tabular LOSO XGBoost ({self.model_path})")
        if self.ensemble is not None:
            print("Model ENSEMBLE: tuned dasheng + digest stacker (action 'analyze')")
        if self.jepa is not None:
            print(f"Model JEPA: V-JEPA 2.1 ViT-L LoRA x{len(self.jepa.folds)} folds (action 'jepa_video')")
        print(f"Hardware Acceleration: {self.active_device}")
        print(f"Ready for connections from localhost and remote machines...")
        print(f"============================================================")
        sys.stdout.flush()

        def _serve():
            while self.running:
                try:
                    readable, _, _ = select.select([self.server_socket], [], [], 0.5)
                    if readable:
                        client_sock, client_addr = self.server_socket.accept()
                        t = threading.Thread(
                            target=self.handle_client,
                            args=(client_sock, client_addr),
                            daemon=True
                        )
                        t.start()
                except Exception:
                    if not self.running:
                        break

        if block:
            _serve()
        else:
            self.listen_thread = threading.Thread(target=_serve, daemon=True)
            self.listen_thread.start()

    def stop(self):
        """Stops the socket server."""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2.0)
        print("[V2 Socket Server] Server stopped.")


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, device: str = "cuda", enable_jepa: bool = True,
               enable_webcam_pilot: bool = False):
    """CLI runner."""
    server = V2ModelServer(host=host, port=port, device=device, enable_jepa=enable_jepa,
                           enable_webcam_pilot=enable_webcam_pilot)
    try:
        server.start(block=True)
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
    finally:
        server.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeceptionX V2 Model Socket Service")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host interface to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference compute device (default: cuda)")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, device=args.device)

