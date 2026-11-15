from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from pathlib import Path
from urllib.parse import urlparse,parse_qs

def make_server(pipeline, port=8765):
    from .showcase import ShowcaseSession
    showcase=getattr(pipeline,"showcase",None)
    if showcase is None:
        showcase=ShowcaseSession(pipeline)
        pipeline.showcase=showcase
    from .dataset_recorder import DatasetRecorder
    dataset=getattr(pipeline,"dataset_recorder",None)
    if dataset is None:
        dataset=DatasetRecorder(pipeline);pipeline.dataset_recorder=dataset
    if getattr(pipeline,"heart_rate",None) is None:
        hr_config=dict(pipeline.config.get("heart_rate") or {})
        faces=pipeline.config.get("local_ensemble") or {}
        hr_config.setdefault("face_model_path",faces.get("face_model_path"))
        hr_config.setdefault("expected_face_sha256",faces.get("expected_face_sha256"))
        if hr_config.get("enabled",True):
            from .capture import add_frame_tap
            from .rppg import HeartRateMonitor
            pipeline.heart_rate=HeartRateMonitor(hr_config)
            add_frame_tap(pipeline.heart_rate)
    action_lock=threading.RLock()
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)
        def log_message(self,*args): pass
        def send_json(self,status,value):
            data = json.dumps(value,allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers(); self.wfile.write(data)
        def allowed(self):
            expected = {f"127.0.0.1:{self.server.server_port}",f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in expected: return False
            origin = self.headers.get("Origin")
            if origin:
                parsed = urlparse(origin)
                if parsed.scheme!="http" or parsed.netloc not in expected: return False
            return True
        def do_GET(self):
            if not self.allowed(): return self.send_json(403,{"error":"Loopback same-origin only"})
            if self.path=="/api/dataset": return self.send_json(200,dataset.snapshot())
            if urlparse(self.path).path=="/api/dataset/preview":
                try:data=dataset.preview(parse_qs(urlparse(self.path).query).get("take_id",[None])[0])
                except ValueError as exc:return self.send_json(404,{"error":str(exc)})
                self.send_response(200);self.send_header("Content-Type","video/mp4")
                self.send_header("Content-Length",str(len(data)));self.send_header("Cache-Control","no-store")
                self.send_header("X-Content-Type-Options","nosniff");self.end_headers();self.wfile.write(data);return
            if self.path=="/api/state": return self.send_json(200,pipeline.state())
            if self.path=="/api/claim-inputs": return self.send_json(200,pipeline.claim_input_preview())
            if self.path=="/api/showcase": return self.send_json(200,showcase.snapshot())
            if self.path=="/api/heart-rate":
                monitor=getattr(pipeline,"heart_rate",None)
                return self.send_json(200,monitor.snapshot() if monitor is not None else {"status":"disabled","display_only":True})
            if self.path=="/api/health": return self.send_json(200,{"status":"ok","mode":pipeline.mode})
            pages={"/":"showcase.html","/showcase":"showcase.html","/showcase.html":"showcase.html",
                   "/dashboard":"index.html","/index.html":"index.html"}
            if self.path not in pages:return self.send_json(404,{"error":"Not found"})
            data = Path(__file__).with_name("static").joinpath(pages[self.path]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Content-Length",str(len(data)))
            self.send_header("Content-Security-Policy","default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; media-src 'self'; frame-ancestors 'none'")
            self.send_header("Referrer-Policy","no-referrer")
            self.end_headers(); self.wfile.write(data)
        def do_POST(self):
            if not self.allowed(): return self.send_json(403,{"error":"Loopback same-origin only"})
            if self.headers.get("Content-Type","").split(";")[0]!="application/json":
                return self.send_json(415,{"error":"application/json required"})
            try:
                size = int(self.headers.get("Content-Length","0"))
                if size<0 or size>32768: return self.send_json(413,{"error":"Request too large"})
                body = json.loads(self.rfile.read(size) or b"{}")
                if not isinstance(body,dict): raise ValueError("Expected JSON object")
                with action_lock:
                    if self.path=="/api/dataset/start":value=dataset.start(body.get("participant_id"))
                    elif self.path=="/api/dataset/save":value=dataset.save(body.get("take_id"),body.get("label"),body.get("statement",""))
                    elif self.path=="/api/dataset/discard":value=dataset.cancel()
                    elif self.path=="/api/stop" and dataset.is_active():value=dataset.cancel()
                    elif dataset.is_active():raise RuntimeError("Dataset recording owns capture. Label or discard it before using other controls.")
                    elif self.path=="/api/audio/reconnect":
                        if showcase.is_active():raise RuntimeError("Cancel the active showcase before reconnecting the microphone")
                        with pipeline.control_lock:
                            state=pipeline.state()
                            if (state.get("answer",{}).get("status") in ("capturing","finishing","processing")
                                    or any(v.get("status") in ("starting","running","stopping") for v in state.get("capture",{}).values())
                                    or any(t.is_alive() for t in pipeline.threads)):
                                raise RuntimeError("Stop capture before reconnecting the microphone")
                            from .capture import reconnect_microphone
                            value=reconnect_microphone(pipeline.config.get("audio_device"))
                            dataset.clear_capture_error(value.get("message","Microphone reconnected. Try Add new data again."))
                    elif self.path=="/api/showcase/startwarmup":value=showcase.start_warmup()
                    elif self.path=="/api/showcase/answer":value=showcase.start_answer(visual_claim_check=showcase.claim_check_enabled())
                    elif self.path=="/api/showcase/prepare":value=showcase.prepare_models()
                    elif self.path=="/api/showcase/video":value=showcase.configure_video(body.get("enabled"))
                    elif self.path=="/api/showcase/claim-check":value=showcase.configure_claim_check(body.get("enabled"))
                    elif self.path=="/api/showcase/reset":value=showcase.reset()
                    elif self.path=="/api/showcase/cancel":value=showcase.cancel()
                    elif self.path=="/api/stop":value=showcase.cancel() if showcase.is_active() else pipeline.stop()
                    else:
                        if showcase.is_active():raise RuntimeError("The showcase is active. Cancel it before using dashboard controls.")
                        if self.path=="/api/session": value = pipeline.start(body.get("mode"))
                        elif self.path=="/api/prepare": value = pipeline.prepare_runtime()
                        elif self.path=="/api/answer/finish": value = pipeline.finish_answer(stop_now=body.get("stop_now",False))
                        elif self.path=="/api/transcript": value = pipeline.set_transcript(body.get("text"))
                        elif self.path=="/api/calibrate": value = pipeline.calibrate()
                        else: return self.send_json(404,{"error":"Not found"})
                return self.send_json(200,value)
            except (ValueError,RuntimeError) as exc: self.send_json(400,{"error":str(exc)})
            except Exception as exc: self.send_json(500,{"error":type(exc).__name__})
    server = ThreadingHTTPServer(("127.0.0.1",port),Handler)
    server.daemon_threads = True
    return server
