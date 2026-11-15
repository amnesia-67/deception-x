"""Capture orchestration and a single inference worker with latest-window backpressure."""
from collections import deque
import copy
import math
import queue
try:
    import resource
except ImportError:
    resource = None
import threading
import time
import uuid
import numpy as np
from .temporal import VideoFrame, AudioChunk, WindowBuffer, AudioBuffer
from .answer_pipeline import AnswerPipelineMixin
from .claim_pipeline import ClaimPipelineMixin

DEFAULTS = {"capture_fps":30, "sample_fps":24, "frames_per_window":64,
            "hop_seconds":32/24, "models":{}}

def public(value):
    """No embeddings, images, waveforms or nonfinite values in state snapshots."""
    if isinstance(value, dict): return {k:public(v) for k,v in value.items() if not k.startswith("_")}
    if isinstance(value, (list,tuple)): return [public(v) for v in value]
    if isinstance(value, np.ndarray): return {"shape":list(value.shape), "redacted":True}
    if isinstance(value, np.generic): return public(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value

def peak_rss_mb():
    if resource is not None:
        import platform
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024 if platform.system()=="Darwin" else 1024)
    import psutil
    info = psutil.Process().memory_info()
    return getattr(info,"peak_wset",info.rss)/(1024*1024)

class Pipeline(AnswerPipelineMixin,ClaimPipelineMixin):
    def __init__(self, config=None):
        self.config = {**DEFAULTS, **(config or {})}
        self.config.pop("manager", None)  # Ignore removed legacy explainer settings.
        # Validate before creating any threads/devices.
        WindowBuffer(self.config["sample_fps"],self.config["frames_per_window"],self.config["hop_seconds"])
        self.lock = threading.RLock()
        self.control_lock = threading.Lock()
        self.mode = "stopped"
        self.session_id = None
        self.created = time.monotonic()
        self.latest = None
        self.errors = deque(maxlen=15)
        self.capture = None
        self.stop_event = threading.Event()
        self.work = queue.Queue(maxsize=1)
        self.threads = []
        self.transcript = {"text":"", "status":"missing", "source":"manual", "updated_at":None}
        self.calibration = {"status":"not_started", "reason":"Diagnostic baseline only; physiological feature models are missing."}
        self.calibration_values = []
        self.metrics = {}
        self.last_seen_sequences = deque(maxlen=2000)
        self.last_processed_sequence = -1
        self.completed_times = deque(maxlen=60)
        self.answer = {}
        self.claim_runtime = None
        self._claim_input_preview = None
        self.video_client = None
        self.fast_runtime = None
        self.prepare_thread = None
        self.closed = False
        self.runtime_preparation = {"status":"not_prepared", "components":{},
                                    "local_weights_loaded":False, "kernels_warmed":False}

    def _runtime_progress(self, progress):
        with self.lock:
            if not self.closed:
                self.runtime_preparation["progress"] = dict(progress)

    def prepare_runtime(self):
        """Prepare resident weights without sensors; split mode also checks test video."""
        if self.config.get("runtime_mode")=="split_local_video":return self._prepare_split_runtime()
        with self.control_lock:
            if self.closed:
                raise RuntimeError("Pipeline is closing; restart the application before preparing models")
            if self.config.get("runtime_mode") != "remote_fast":
                raise RuntimeError("Resident preparation requires the remote_fast profile; baseline inference owns its own model lifecycle")
            if self.mode != "stopped":
                raise RuntimeError("Stop the session before preparing local models")
            if self.prepare_thread and self.prepare_thread.is_alive():
                return self.state()["runtime"]
            if self.fast_runtime is None:
                from .fast_runtime import FastRuntime
                self.fast_runtime = FastRuntime(self.config, on_progress=self._runtime_progress)
            if self.fast_runtime.snapshot()["active_jobs"]:
                raise RuntimeError("Previous model work is still finishing; wait before preparation")
            with self.lock:
                self.runtime_preparation = {"status":"preparing", "components":{},
                                            "local_weights_loaded":False, "kernels_warmed":False}
            def prepare():
                try:
                    value = self.fast_runtime.prepare()
                except Exception as exc:
                    value = {"status":"error", "reason":type(exc).__name__+": "+str(exc)[:300],
                             "components":{}, "local_weights_loaded":False, "kernels_warmed":False}
                with self.lock:
                    if not self.closed:
                        self.runtime_preparation = value
            self.prepare_thread = threading.Thread(target=prepare, name="prepare-local-models", daemon=True)
            self.prepare_thread.start()
        return self.state()["runtime"]

    def _runtime_state(self):
        if self.config.get("runtime_mode")=="split_local_video":return self._split_runtime_state()
        value = copy.deepcopy(self.runtime_preparation)
        remote = self.config.get("remote_vision", {})
        value.update(mode=self.config.get("runtime_mode", "local_baseline"),
                     remote={"status":remote.get("status", "not_configured"),
                             "host":remote.get("host"), "port":remote.get("port",5002),
                             "reason":remote.get("reason", "Reviewed raw-video service has not been configured."),
                             "reachable":remote.get("reachable"),
                             "health_elapsed_ms":remote.get("health_elapsed_ms"),
                             "advertised_models":remote.get("advertised_models",[]),
                             "advertised_video_model":remote.get("advertised_video_model"),
                             "encoder_verified":False},
                     live_ready=False,
                     reason="Live fast inference is unavailable: "+(
                         self.config.get("deployment",{}).get("runtime_blocker") or
                         remote.get("reason") or "Matching model exports and a reviewed raw-video service are required."))
        if self.fast_runtime:
            value["lifecycle"] = self.fast_runtime.snapshot()
        return value

    def clear_session_data(self):
        """Forget a completed participant's in-memory data while retaining models."""
        with self.control_lock:
            if self.capture and any(self.capture.snapshot().get(k,{}).get("status") in
                    ("starting","running","stopping") for k in ("camera","audio")):
                raise RuntimeError("Stop capture before clearing the participant session")
            if any(t.is_alive() for t in self.threads):
                raise RuntimeError("Previous answer is still finishing")
            if self.fast_runtime and (self.fast_runtime.snapshot().get("active_jobs") or
                    self.fast_runtime.snapshot().get("active_prepares")):
                raise RuntimeError("Previous model work is still finishing")
            with self.lock:
                self.stop_event.set();self.mode="stopped";self.session_id=None
                self.created=time.monotonic();self.latest=None;self.answer={}
                self._claim_input_preview=None
                self.transcript={"text":"","status":"missing","source":"manual","updated_at":None}
                self.metrics={};self.errors.clear()
                self.calibration={"status":"not_started","reason":"Diagnostic baseline only; physiological feature models are missing."}
                self.calibration_values=[];self.last_seen_sequences.clear();self.completed_times.clear()
                self.last_processed_sequence=-1
                for name,field in (("video","frames"),("audio","chunks")):
                    value=getattr(self,name,None)
                    if value is not None:getattr(value,field).clear()
                if hasattr(self,"answer_frames"):self.answer_frames.clear()
                monitor=getattr(self,"heart_rate",None)
                if monitor is not None:monitor.clear_recorded_reading()
                while True:
                    try:self.work.get_nowait()
                    except queue.Empty:break
        return self.state()

    def close(self):
        """Stop capture and release resident models after any active work ends."""
        dataset=getattr(self,"dataset_recorder",None)
        if dataset is not None:dataset.close()
        monitor=getattr(self,"heart_rate",None)
        if monitor is not None:
            from .capture import remove_frame_tap
            remove_frame_tap(monitor);monitor.close()
        showcase=getattr(self,"showcase",None)
        if showcase is not None:showcase.close()
        self.stop()
        if self.claim_runtime is not None:self.claim_runtime.close()
        with self.control_lock:
            with self.lock:
                self.closed = True
            if self.video_client:self.video_client.close()
            if self.fast_runtime:
                return self.fast_runtime.close()
        return {"status":"closed"}

    def start(self, mode, *, auto_stop_seconds=None, visual_claim_check=False):
        if type(visual_claim_check) is not bool:
            raise ValueError("Claim image authorization must be boolean")
        if mode not in ("demo","live"): raise ValueError("Mode must be demo or live")
        if mode=="live" and self.config.get("runtime_mode")=="split_local_video":return self._start_answer(auto_stop_seconds=auto_stop_seconds,visual_claim_check=visual_claim_check)
        if auto_stop_seconds is not None:raise ValueError("Automatic answer timing requires split_local_video mode")
        with self.control_lock:
            if self.closed:
                raise RuntimeError("Pipeline is closing; restart the application")
            if mode == "live" and self.config.get("runtime_mode") == "remote_fast":
                # A config flag or a pong is never proof of a working encoder.
                # Enabling a new service requires source/protocol/fixture review.
                raise RuntimeError(self._runtime_state()["reason"])
            self._stop()
            if any(t.is_alive() for t in self.threads):
                raise RuntimeError("Previous inference is still stopping; wait before starting another session")
            with self.lock:
                self.mode, self.session_id = mode, uuid.uuid4().hex[:12]
                self.created = time.monotonic()
                self.latest = None
                self.errors.clear()
                self.transcript = {"text":"", "status":"missing", "source":"manual", "updated_at":None}
                self.calibration = {"status":"not_started", "reason":"Diagnostic baseline only; physiological feature models are missing."}
                self.calibration_values = []
                self.last_seen_sequences.clear()
                self.last_processed_sequence = -1
                self.completed_times.clear()
                self.metrics = {"completed_windows":0,"dropped_windows":0,"rejected_windows":0,"queue_depth":0,
                                "window_latency_ms":None,"process_peak_rss_mb":peak_rss_mb(),
                                "unique_frames_per_second":0,"inference_updates_per_second":0}
                self.stop_event = threading.Event()
                self.work = queue.Queue(maxsize=1)
                self.video = WindowBuffer(self.config["sample_fps"],64,self.config["hop_seconds"])
                self.audio = AudioBuffer()
                if mode == "live":
                    from .capture import LiveCapture
                    self.capture = LiveCapture(camera_index=self.config.get("camera_device",0),
                                               audio_device=self.config.get("audio_device"),
                                               camera_fps=self.config["capture_fps"])
                    self.capture.start()
                else: self.capture = None
                self.threads = [threading.Thread(target=self._ingest, name="capture-ingest",daemon=True),
                                threading.Thread(target=self._infer, name="model-worker",daemon=True)]
                for t in self.threads: t.start()
        return self.state()

    def stop(self):
        with self.control_lock: self._stop()
        return self.state()

    def _stop(self):
        self.stop_event.set()
        with self.lock:self._claim_input_preview=None
        if self.answer:
            if self.video_client and (self.answer.get("status")=="processing" or self.video_client.snapshot().get("busy")):
                self.video_client.close()
            if self.answer.get("status")!="complete":self.answer["status"]="stopped"
        if self.capture: self.capture.stop()
        # MPS work cannot be cancelled safely mid-kernel. Do not allow a second worker.
        for t in self.threads: t.join(timeout=2)
        with self.lock:
            self.mode = "stopped"
            if hasattr(self,"video"): self.video.frames.clear()
            if hasattr(self,"audio"): self.audio.chunks.clear()
            if hasattr(self,"answer_frames"): self.answer_frames.clear()
            while True:
                try: self.work.get_nowait()
                except queue.Empty: break

    def _ingest(self):
        epoch = time.monotonic()
        seq = 0
        next_demo = epoch
        pending = deque(maxlen=2)
        while not self.stop_event.is_set():
            try:
                if self.mode == "demo":
                    now = time.monotonic()
                    frames, chunks = [], []
                    # These are in-memory non-person synthetic test signals, never recorded.
                    while next_demo <= now and len(frames)<10:
                        image = np.zeros((96,128,3),np.uint8)
                        image[:, :, 2] = 25
                        image[20:60,(seq*3)%90:(seq*3)%90+20,0] = 180
                        frames.append(VideoFrame(next_demo,image,seq))
                        sample_start = round(seq*16000/self.config["capture_fps"])
                        sample_end = round((seq+1)*16000/self.config["capture_fps"])
                        a = np.arange(sample_start,sample_end,dtype=np.float32)
                        chunks.append(AudioChunk(epoch+sample_start/16000,16000,.02*np.sin(2*np.pi*220*a/16000)))
                        next_demo += 1/self.config["capture_fps"]
                        seq += 1
                else:
                    frames, chunks = self.capture.drain()
                for frame in frames: self.video.add(frame)
                for chunk in chunks: self.audio.add(chunk)
                for window in self.video.ready():
                    if len(pending)==pending.maxlen:
                        self.metrics["dropped_windows"] += 1
                    pending.append((window,time.monotonic()))
                while pending:
                    window,ready_at = pending[0]
                    waveform, audio_meta = self.audio.slice(window.start,window.end)
                    # Wait for callback/resampler lag without manufacturing missing samples.
                    if waveform is None and time.monotonic()-ready_at < .25: break
                    pending.popleft()
                    audio_meta["alignment_wait_ms"] = (time.monotonic()-ready_at)*1000
                    item = (window,waveform,audio_meta,time.monotonic())
                    try: self.work.put_nowait(item)
                    except queue.Full:
                        try: self.work.get_nowait()
                        except queue.Empty: pass
                        self.metrics["dropped_windows"] += 1
                        try: self.work.put_nowait(item)
                        except queue.Full: self.metrics["dropped_windows"] += 1
                self.metrics["rejected_windows"] = self.video.rejected
                self.metrics["queue_depth"] = self.work.qsize()
            except Exception as exc:
                self.errors.append("Capture processing failed: "+type(exc).__name__+": "+str(exc)[:180])
            self.stop_event.wait(.01)

    def _infer(self):
        from .models import ModelRegistry,input_settings
        from .features import signal_diagnostics
        registry = ModelRegistry(self.config)
        def mark_stage(name):
            with self.lock:
                if not self.stop_event.is_set(): self.metrics["processing_stage"] = name
        registry.on_stage = mark_stage
        while not self.stop_event.is_set():
            try: window,waveform,audio_meta,enqueued = self.work.get(timeout=.2)
            except queue.Empty: continue
            started = time.monotonic()
            try:
                clip = np.stack([f.rgb for f in window.frames])
                with self.lock: transcript = copy.deepcopy(self.transcript)
                runtime_result = {}
                if self.mode != "demo" and self.config.get("runtime_mode") == "remote_fast":
                    if self.fast_runtime is None:
                        raise RuntimeError("Prepare local models before fast inference")
                    runtime_result = self.fast_runtime.infer(clip,waveform,transcript,window.id,self.stop_event)
                    branches, transcript = runtime_result["branches"], runtime_result["transcript"]
                    if transcript.get("source") == "local_asr":
                        transcript.update(scope="video_window",start=window.start,end=window.end)
                else:
                    if self.mode!="demo" and waveform is not None and self.config.get("models",{}).get("asr",{}).get("path"):
                        mark_stage("local_transcription")
                        transcript = registry.transcribe(waveform)
                        transcript.update(source="local_asr",scope="video_window",start=window.start,end=window.end)
                    if self.stop_event.is_set(): break
                    complete_text = transcript.get("text","") if transcript.get("status") in ("available","base") else ""
                    branches = registry.run(clip,waveform,complete_text,demo=self.mode=="demo")
                if self.stop_event.is_set(): break
                features = signal_diagnostics(clip,waveform,16000)
                mark_stage("fusion")
                from .heads import run_fusion,run_tabular,run_stack
                head_config = {**self.config,"runtime":{"text_scope":"video_window" if transcript.get("scope")=="video_window" else "session_context"}}
                if self.config.get("models",{}).get("fusion",{}).get("architecture")=="friend-reda-dare-synthetic-v1":
                    from .teammate_fusion import run_teammate_fusion
                    branches["fusion"] = run_teammate_fusion(branches,head_config)
                else:
                    branches["fusion"] = run_fusion(branches,head_config)
                branches["tabular"] = run_tabular(features,head_config)
                branches["stack"] = run_stack(branches,head_config)
                finished = time.monotonic()
                # A stopped session must not publish a late result.
                if self.stop_event.is_set(): break
                stamps = [(finished,f.sequence) for f in window.frames if f.sequence>self.last_processed_sequence]
                with self.lock:
                    if self.stop_event.is_set() or self.mode=="stopped": break
                    self.last_seen_sequences.extend(stamps)
                    self.last_processed_sequence = max(f.sequence for f in window.frames)
                    self.completed_times.append(finished)
                    while self.last_seen_sequences and self.last_seen_sequences[0][0]<finished-30:
                        self.last_seen_sequences.popleft()
                    times = [t for t in self.completed_times if t>=finished-30]
                    elapsed = max(.001,min(30,finished-self.created))
                    self.metrics.update(completed_windows=self.metrics["completed_windows"]+1,
                                        window_latency_ms=(finished-started)*1000,
                                        queue_wait_ms=(started-enqueued)*1000,
                                        result_age_seconds=finished-window.end,
                                        unique_frames_per_second=len(set(s for t,s in self.last_seen_sequences))/elapsed,
                                        inference_updates_per_second=len(times)/elapsed,
                                        measurement_window_seconds=elapsed,
                                        process_peak_rss_mb=peak_rss_mb())
                    self.latest = {"window":{**window.metadata(),"audio":audio_meta},
                                   "branches":public(branches),"features":public(features),"transcript":transcript,
                                   "runtime_timings":runtime_result.get("timings"),
                                   "produced_at":finished,
                                   "preprocessing_settings":input_settings(self.config),
                                   "calibrated_task_output":None,"mock":self.mode=="demo"}
                    self.metrics["processing_stage"] = "idle"
                    if self.calibration.get("status")=="collecting":
                        self.calibration_values.append(public(features))
                        if finished-self.calibration["start"]>=15:
                            self.calibration = {"status":"complete","windows":len(self.calibration_values),
                                                "reason":"Diagnostic baseline captured; no validated physiology, truth labels or calibration parameters."}
                    future = runtime_result.get("_emotion_future")
                    if future is not None:
                        self._watch_emotion(future,self.session_id,window.id,self.stop_event)
            except Exception as exc:
                with self.lock: self.errors.append("Inference failed: "+type(exc).__name__+": "+str(exc)[:250])

    def _watch_emotion(self, future, session_id, window_id, source_stop):
        """A supplemental result can update only its own still-current window."""
        def complete(done):
            try:
                item = done.result()
            except Exception:
                return
            with self.lock:
                if (source_stop.is_set() or self.closed or self.mode == "stopped" or
                        self.session_id != session_id or not self.latest or
                        self.latest.get("window",{}).get("id") != window_id or
                        item.get("source_window_id") != window_id or
                        item.get("status") == "cancelled"):
                    return
                self.latest["branches"]["emotion"] = public(item)
        future.add_done_callback(complete)


    def state(self):
        recorder=getattr(self,"dataset_recorder",None)
        dataset=recorder.snapshot() if recorder is not None else None
        monitor=getattr(self,"heart_rate",None)
        # Display-only rPPG readout. It is reported beside capture state and is
        # never read by fusion, stack, tabular or showcase scoring.
        heart_rate=monitor.snapshot() if monitor is not None else None
        with self.lock:
            metrics = dict(self.metrics)
            end = self.latest.get("window",{}).get("end") if self.latest else None
            metrics["current_result_age_seconds"] = max(0,time.monotonic()-end) if end is not None else None
            if self.capture: capture = self.capture.snapshot()
            elif self.mode == "demo": capture = {"camera":{"status":"mock","observed_fps":self.config["capture_fps"]},"audio":{"status":"mock","sample_rate":16000}}
            else: capture = {"camera":{"status":"stopped"},"audio":{"status":"stopped"}}
            if dataset and dataset.get("active") and dataset.get("capture"):capture=dataset["capture"]
            value = {"mode":self.mode,"session_id":self.session_id,"uptime_seconds":time.monotonic()-self.created,
                     "config":{k:self.config[k] for k in ("capture_fps","sample_fps","frames_per_window","hop_seconds")},
                     "dataset":dataset,"capture":capture,"heart_rate":heart_rate,"metrics":metrics,"latest":self.latest,"answer":self.answer,"errors":list(self.errors),
                     "calibration":self.calibration,"transcript":self.transcript,"runtime":self._runtime_state(),
                     "deployment":{**self.config.get("deployment",{}),"fusion_architecture":self.config.get("models",{}).get("fusion",{}).get("architecture"),
                                   "fusion_training_domain":"generated_synthetic_features" if self.config.get("models",{}).get("fusion",{}).get("architecture")=="friend-reda-dare-synthetic-v1" else None,
                                   "pilot_blend_weights":(self.config.get("pilot_blend") or {}).get("weights"),
                                   "local_asr_configured":bool(self.config.get("models",{}).get("asr",{}).get("path"))},
                     "warning":"Experimental observations; task-specific trained heads and calibration are absent."}
            return public(copy.deepcopy(value))

    def set_transcript(self,text):
        if not isinstance(text,str) or len(text)>16000: raise ValueError("Transcript limit is 16000 characters")
        with self.lock:
            self.transcript = {"text":text,"status":"available" if text else "missing","source":"manual",
                               "updated_at":time.monotonic(),"scope":"session_context; not word-aligned to video"}
        return self.state()

    def calibrate(self):
        with self.lock:
            if self.mode=="stopped": raise ValueError("Start a session first")
            self.calibration_values = []
            self.calibration = {"status":"collecting","start":time.monotonic(),"duration_seconds":15,
                                "reason":"Diagnostic baseline only; not a physiological or deception calibration."}
        return self.state()

