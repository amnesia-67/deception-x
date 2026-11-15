"""
DeceptionX V2 Model Socket Client Helper & Verification Tool.
Accredited to Gemini.pdf Specifications.

Connects to the V2 Model Socket Server over localhost or network IP,
sends feature vectors, and parses the returned prediction, probability,
and TreeSHAP forensic explanations.
"""

import socket
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Union


class V2ModelSocketClient:
    """Client for communicating with the V2 Model TCP Socket."""
    def __init__(self, host: str = "127.0.0.1", port: int = 5002, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def analyze(self, frames, audio_16k_mono, transcript: str, fps: float = None, timestamps=None,
                quality: int = 90, request_id: str = None, source_id: str = None) -> Dict[str, Any]:
        """Best model (stacked ensemble). One spoken answer:
        frames: RGB uint8 arrays (or JPEG paths) covering the answer, raw camera frames are fine (the server
                crops the face); fps or per-frame timestamps (seconds) are required.
        audio_16k_mono: float array in [-1, 1] at 16 kHz, mono, same span as the frames.
        transcript: verbatim text of the answer (e.g. from faster-whisper on Node 1).
        request_id / source_id: optional identifiers (<= 128 chars) echoed back in the reply, so a reply can
                be matched to the capture that produced it; check them before displaying a result.
        """
        import base64

        enc = self._encode_frames(frames, quality)
        # float32, not int16: quantisation changes the silence floor that the dB-scale mel front-end and the
        # pause/energy features depend on (training audio was decoded to float).
        a = np.asarray(audio_16k_mono, dtype="<f4").reshape(-1)
        payload = {"action": "analyze", "frames": enc, "transcript": transcript,
                   "audio_f32_b64": base64.b64encode(a.tobytes()).decode()}
        for k, v in (("request_id", request_id), ("source_id", source_id)):
            if v is not None:
                payload[k] = v
        if timestamps is not None:
            payload["timestamps"] = [float(t) for t in timestamps]
        elif fps:
            payload["fps"] = float(fps)
        else:
            raise ValueError("pass fps or timestamps")
        prev, self.timeout = self.timeout, max(self.timeout, 180.0)
        try:
            return self._send_request(payload)
        finally:
            self.timeout = prev

    @staticmethod
    def _encode_frames(frames, quality: int = 90):
        import base64
        import cv2

        enc = []
        for f in frames:
            if isinstance(f, str):
                with open(f, "rb") as fh:
                    enc.append(base64.b64encode(fh.read()).decode())
            else:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(f, np.uint8), cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    raise ValueError("JPEG encoding failed")
                enc.append(base64.b64encode(buf.tobytes()).decode())
        return enc

    def predict_video(self, frames, fps: float = None, timestamps=None, quality: int = 90,
                      request_id: str = None, source_id: str = None) -> Dict[str, Any]:
        """Send one answer's raw camera frames to the V-JEPA 2.1 ViT-L classifier (uncalibrated score).

        frames: image file paths or RGB uint8 arrays (H, W, 3), >= 16 frames spanning the answer. The server
        selects 128 frames by time and crops the face, exactly as in training, so fps or per-frame
        timestamps (seconds) are required. request_id / source_id are echoed back.
        """
        enc = self._encode_frames(frames, quality)
        payload = {"action": "jepa_video", "frames": enc}
        for k, v in (("request_id", request_id), ("source_id", source_id)):
            if v is not None:
                payload[k] = v
        if timestamps is not None:
            payload["timestamps"] = [float(t) for t in timestamps]
        elif fps:
            payload["fps"] = float(fps)
        else:
            raise ValueError("pass fps or timestamps")
        prev, self.timeout = self.timeout, max(self.timeout, 180.0)  # ViT-L video inference takes seconds
        try:
            return self._send_request(payload)
        finally:
            self.timeout = prev

    def ping(self) -> Dict[str, Any]:
        """Sends health-check ping to V2 socket server."""
        payload = {"action": "ping"}
        return self._send_request(payload)

    def predict(
        self,
        features: Union[List[float], np.ndarray],
        normalize: bool = False
    ) -> Dict[str, Any]:
        """
        Sends one feature vector (order: the server's `schema` action) to the V2 model socket.
        Returns model output dictionary.
        """
        if isinstance(features, np.ndarray):
            feat_list = features.reshape(-1).tolist()
        else:
            feat_list = list(features)


        payload = {
            "z": feat_list,
            "normalize": normalize
        }
        return self._send_request(payload)

    def predict_batch(
        self,
        batch_features: Union[List[List[float]], np.ndarray],
        normalize: bool = False
    ) -> Dict[str, Any]:
        """Sends batch of feature vectors (N, 250)."""
        if isinstance(batch_features, np.ndarray):
            feat_list = batch_features.tolist()
        else:
            feat_list = [list(row) for row in batch_features]

        payload = {
            "batch": feat_list,
            "normalize": normalize
        }
        return self._send_request(payload)

    def predict_jepa(
        self,
        f_v: Optional[Union[List, np.ndarray]] = None,
        f_a: Optional[Union[List, np.ndarray]] = None,
        f_t: Optional[Union[List, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Sends JEPA request to socket server for Branch 1 neural discrepancy inference.
        """
        payload: Dict[str, Any] = {"action": "jepa"}
        if f_v is not None:
            payload["f_v"] = f_v.tolist() if isinstance(f_v, np.ndarray) else f_v
        if f_a is not None:
            payload["f_a"] = f_a.tolist() if isinstance(f_a, np.ndarray) else f_a
        if f_t is not None:
            payload["f_t"] = f_t.tolist() if isinstance(f_t, np.ndarray) else f_t
        return self._send_request(payload)

    def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sends JSON request over socket and reads JSON response."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect((self.host, self.port))
            req_data = json.dumps(payload).encode("utf-8") + b"\n"
            sock.sendall(req_data)

            # Read response
            resp_bytes = b""
            while b"\n" not in resp_bytes:
                chunk = sock.recv(16384)
                if not chunk:
                    break
                resp_bytes += chunk

            line = resp_bytes.decode("utf-8").strip()
            if not line:
                raise ConnectionError("Empty response from V2 model socket.")
            return json.loads(line)
        finally:
            sock.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query DeceptionX V2 Model Socket Server")
    parser.add_argument("--host", default="127.0.0.1", help="Target host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5002, help="Target port (default: 5002)")
    parser.add_argument("--ping", action="store_true", help="Send ping test")
    parser.add_argument("--jepa", action="store_true", help="Test Branch 1 JEPA neural discrepancy model")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic 250-dim vector and predict")
    args = parser.parse_args()

    client = V2ModelSocketClient(host=args.host, port=args.port)

    if args.ping:
        res = client.ping()
        print("Ping Response:", json.dumps(res, indent=2))
        return

    if args.jepa:
        print(f"Connecting to Socket on {args.host}:{args.port} for JEPA model test...")
        t0 = time.perf_counter()
        res = client.predict_jepa()
        roundtrip_ms = (time.perf_counter() - t0) * 1000.0
        print(f"\n[JEPA Model Socket Response] (Roundtrip: {roundtrip_ms:.2f} ms):")
        print(json.dumps(res, indent=2))
        return

    # Default: run test inference
    np.random.seed(42)
    dummy_z = np.random.randn(250).astype(np.float32)
    print(f"Connecting to V2 Model Socket on {args.host}:{args.port}...")
    t0 = time.perf_counter()
    res = client.predict(dummy_z)
    roundtrip_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n[V2 Model Socket Response] (Roundtrip: {roundtrip_ms:.2f} ms):")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
