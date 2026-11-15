"""Whole-answer controls for the local-audio/text, remote-video profile."""
from collections import deque
import copy
import math
import queue
import threading
import time
import uuid
import numpy as np
from .answer_capture import select_answer
from .temporal import AudioBuffer


class AnswerPipelineMixin:
    def set_remote_video(self, enabled):
        if type(enabled) is not bool:raise ValueError("enabled must be a boolean")
        with self.control_lock:
            if self.closed:raise RuntimeError("Pipeline is closing")
            if self.answer.get("status") in ("capturing","finishing","processing") or self.mode=="demo":
                raise RuntimeError("Stop the current test before changing GPU video")
            if self.prepare_thread and self.prepare_thread.is_alive():raise RuntimeError("Wait for model preparation to finish")
            if self.capture and any(self.capture.snapshot().get(k,{}).get("status") in ("starting","running","stopping") for k in ("camera","audio")):
                raise RuntimeError("Stop capture before changing GPU video")
            if any(t.is_alive() for t in self.threads):raise RuntimeError("Previous answer is still finishing")
            if self.fast_runtime:
                state=self.fast_runtime.snapshot()
                if state.get("active_jobs") or state.get("active_prepares"):raise RuntimeError("Previous model work is still finishing")
            if self.video_client:self.video_client.close()
            self.video_client=None
            self.config.setdefault("remote_vision",{})["enabled"]=enabled
            if self.fast_runtime:
                self.fast_runtime.video_client=None
                self.fast_runtime.config.setdefault("remote_vision",{})["enabled"]=enabled
        return self._split_runtime_state()

    def _prepare_split_runtime(self):
        with self.control_lock:
            if self.closed:raise RuntimeError("Pipeline is closing")
            blocker=self.config.get("deployment",{}).get("runtime_blocker")
            if blocker:
                self.runtime_preparation={"status":"blocked","local_weights_loaded":False,"components":{},"reason":blocker}
                return self.state()["runtime"]
            if self.answer.get("status") in ("capturing","finishing","processing"):
                raise RuntimeError("Finish or stop the current answer before preparing")
            if self.prepare_thread and self.prepare_thread.is_alive():return self.state()["runtime"]
            if self.fast_runtime and self.fast_runtime.snapshot().get("active_jobs"):
                raise RuntimeError("Previous model work is still finishing")
            from .remote_jepa_vitl import RemoteJepa21LClassifier,SOURCE_REVISION
            from .split_runtime import SplitRuntime
            remote=self.config.get("remote_vision",{})
            video_enabled=remote.get("enabled",True)
            if video_enabled and (self.video_client is None or self.video_client.snapshot().get("status")=="closed"):
                self.video_client=RemoteJepa21LClassifier(remote["host"],remote.get("port",5002),
                    reviewed_source_revision=remote.get("reviewed_source_revision"),
                    expected_manifest_sha256=remote.get("expected_manifest_sha256"),
                    participant_media_authorized=bool(remote.get("participant_media_authorized")),
                    timeout_seconds=remote.get("timeout_seconds",20),
                    max_frame_edge=remote.get("max_frame_edge",640),profile=remote.get("profile","served"))
            if self.fast_runtime is None:
                self.fast_runtime=SplitRuntime(self.config,video_client=self.video_client,on_progress=self._runtime_progress)
            else:self.fast_runtime.video_client=self.video_client
            self.runtime_preparation={"status":"preparing","components":{},"local_weights_loaded":False}
            def prepare():
                try:
                    value=self.fast_runtime.prepare()
                    if video_enabled:
                        health=self.video_client.probe()
                        if health.get("classifier_ready"):
                            result=self.video_client.benchmark_synthetic(window_id=0)
                            value["video_check"]={"status":"verified_fixture","elapsed_ms":result["elapsed_ms"],
                                                  "manifest_sha256":result["manifest_sha256"],"participant_data_sent":False}
                        else:value["video_check"]={"status":"unavailable","reason":health.get("reason",health.get("status"))}
                    else:value["video_check"]={"status":"disabled","participant_data_sent":False}
                except Exception as exc:
                    value={"status":"error","components":{},"local_weights_loaded":False,
                           "reason":type(exc).__name__+": "+str(exc)[:350]}
                with self.lock:
                    if not self.closed:self.runtime_preparation=value
            self.prepare_thread=threading.Thread(target=prepare,name="prepare-split-models",daemon=True)
            self.prepare_thread.start()
        return self.state()["runtime"]

    def _split_runtime_state(self):
        value=copy.deepcopy(self.runtime_preparation)
        settings=self.config.get("remote_vision",{})
        video_enabled=settings.get("enabled",True)
        remote=self.video_client.snapshot() if self.video_client else {"status":"not_probed"}
        if not video_enabled:remote={"status":"disabled","classifier_ready":False,"classifier_response_validated":False}
        remote.update(host=settings.get("host"),port=settings.get("port",5002),
                      enabled=video_enabled,video_only=True,audio_transmitted=False,transcript_transmitted=False)
        digest=getattr(self.fast_runtime,"digest_stack",None) if self.fast_runtime else None
        digest_snapshot=getattr(digest,"snapshot",None)
        worker=None
        if callable(digest_snapshot):
            try:worker=digest_snapshot()
            except Exception:worker={"ready":False,"status":"error"}
            value["digest_worker"]=worker
            if worker.get("ready") is not True:
                value["local_weights_loaded"]=False
                if value.get("status")!="preparing":
                    value["status"]="partial"
                    value["reason"]="The local digest worker is not ready. Prepare local models again before starting an answer."
        ready=bool(value.get("local_weights_loaded") and (not video_enabled or (remote.get("classifier_ready") and
                   remote.get("classifier_response_validated") and remote.get("participant_media_authorized"))))
        blend_weights=getattr(self.fast_runtime,"pilot_blend_weights",None)
        value["pilot_blend_weights"]=blend_weights
        if blend_weights and blend_weights.get("vision",0)>0 and not video_enabled:
            ready=False
            value["reason"]="This manual blend requires GPU video. Enable video or remove the blend setting."
        value.update(mode="split_local_video",remote=remote,remote_video_enabled=video_enabled,live_ready=ready,
                     locality={"video":"friend_gpu" if video_enabled else "local_face_features_only","audio":"mac","transcript":"mac","ensemble":"mac"},
                     reason=("Ready for one answer; only video leaves this Mac." if video_enabled else "Ready for a local-model test. GPU video is off; the contradiction check has its own setting.") if ready else
                            value.get("reason","Prepare local models and verify the video service before capture."))
        if value.get("status")!="preparing":value["status"]="ready" if ready else value.get("status","not_prepared")
        if self.fast_runtime:value["lifecycle"]=self.fast_runtime.snapshot()
        blocker=self.config.get("deployment",{}).get("runtime_blocker")
        if blocker:value.update(status="blocked",live_ready=False,reason=blocker)
        return value

    def _start_answer(self, *, auto_stop_seconds=None, visual_claim_check=False):
        if auto_stop_seconds is not None:
            if (isinstance(auto_stop_seconds,bool) or not isinstance(auto_stop_seconds,(int,float))
                    or not math.isfinite(auto_stop_seconds) or not 5<=auto_stop_seconds<=30):
                raise ValueError("Automatic answer duration must be within5..30seconds")
            auto_stop_seconds=float(auto_stop_seconds)
        from .capture import LiveCapture
        with self.control_lock:
            if self.closed:raise RuntimeError("Pipeline is closing")
            if not self._split_runtime_state()["live_ready"]:raise RuntimeError("Prepare and verify models before starting an answer")
            if self.answer.get("status") in ("capturing","finishing","processing"):
                raise RuntimeError("The current answer is still active")
            if self.capture and any(self.capture.snapshot().get(k,{}).get("status") in ("starting","running","stopping") for k in ("camera","audio")):
                raise RuntimeError("Previous capture is still stopping; wait or restart the application")
            if any(thread.is_alive() for thread in self.threads):
                if self.mode=="demo":self._stop()
                else:raise RuntimeError("Previous answer is still finishing")
            if self.fast_runtime.snapshot().get("active_jobs"):raise RuntimeError("Previous model work is still finishing")
            with self.lock:
                self.mode="live";self.session_id=uuid.uuid4().hex[:12];self.created=time.monotonic()
                self.latest=None;self.errors.clear();self._claim_input_preview=None
                self.transcript={"text":"","status":"missing","source":"manual","updated_at":None}
                self.stop_event=threading.Event();self.answer_finish_event=threading.Event()
                self.answer_stop_at=None;self.work=queue.Queue(maxsize=1)
                self.answer_frames=deque(maxlen=1000);self.audio=AudioBuffer(retention_seconds=35)
                self.audio.chunks=deque(maxlen=1000)
                self.answer={"status":"capturing","started_at":self.created,"min_seconds":5.,"max_seconds":30.,
                             "target_seconds":[5,7],"source_frames":0,"elapsed_seconds":0.,
                             "auto_stop_seconds":auto_stop_seconds,"timer_started_at":None,"deadline_at":None,
                             "visual_claim_check":visual_claim_check}
                self.metrics={"completed_windows":0,"dropped_windows":0,"rejected_windows":0,"queue_depth":0,
                              "unique_frames_per_second":0,"inference_updates_per_second":0,"window_latency_ms":None}
                self.capture=LiveCapture(camera_index=self.config.get("camera_device",0),
                    audio_device=self.config.get("audio_device"),camera_fps=self.config.get("capture_fps",30),
                    max_frame_edge=self.config.get("capture_max_frame_edge",640))
                self.capture.start()
                self.threads=[threading.Thread(target=self._ingest_answer,name="capture-answer",daemon=True),
                              threading.Thread(target=self._infer_answer,name="analyze-answer",daemon=True)]
                for thread in self.threads:thread.start()
        return self.state()

    def finish_answer(self, *, stop_now=False):
        with self.control_lock:
            if self.config.get("runtime_mode")!="split_local_video" or self.answer.get("status")!="capturing":
                raise RuntimeError("No answer is being captured")
            if not isinstance(stop_now,bool):raise ValueError("stop_now must be a boolean")
            if not stop_now and self.answer.get("elapsed_seconds",0)<self.answer.get("min_seconds",5):
                raise RuntimeError("Speak for at least5 seconds before finishing")
            if not stop_now:
                try:
                    select_answer(tuple(self.answer_frames),identifier=1)
                except ValueError as exc:
                    raise RuntimeError(str(exc)) from exc
            self.answer_stop_at=time.monotonic()
            self.answer["status"]="finishing";self.answer_finish_event.set()
        return self.state()

    def _ingest_answer(self):
        source_stop=self.stop_event;session=self.session_id
        try:
            while not source_stop.is_set():
                frames,chunks=self.capture.drain()
                for chunk in chunks:self.audio.add(chunk)
                audio_start=self.audio.chunks[0].start if self.audio.chunks else float("inf")
                for frame in frames:
                    if frame.timestamp>=audio_start and (self.answer_stop_at is None or frame.timestamp<=self.answer_stop_at):
                        self.answer_frames.append(frame)
                device_status=self.capture.snapshot()
                if time.monotonic()-self.created>1:
                    for modality in ("camera","audio"):
                        if device_status[modality].get("status")=="missing":
                            raise RuntimeError(modality+": "+device_status[modality].get("error","Device unavailable"))
                if not self.answer_frames and time.monotonic()-self.created>8:
                    raise RuntimeError("Camera did not produce frames within8seconds")
                if self.answer_frames and time.monotonic()-self.answer_frames[-1].timestamp>2:
                    raise RuntimeError("Camera stopped delivering frames; answer capture was cancelled")
                if time.monotonic()-self.created>38:
                    raise RuntimeError("Answer capture exceeded its wall-clock deadline")
                automatic=self.answer.get("auto_stop_seconds")
                if automatic is not None:
                    if self.answer.get("timer_started_at") is None and self.answer_frames and all(
                            device_status[k].get("status")=="running" for k in ("camera","audio")):
                        first=self.answer_frames[0].timestamp
                        _,first_audio=self.audio.slice(first,first+.001)
                        if first_audio.get("status")=="available":
                            with self.lock:self.answer.update(timer_started_at=first,deadline_at=first+automatic)
                    cutoff=self.answer.get("deadline_at")
                    if cutoff is None and time.monotonic()-self.created>8:
                        raise RuntimeError("Synchronized camera and microphone did not become ready within8seconds")
                    if cutoff is not None and time.monotonic()>=cutoff and not self.answer_finish_event.is_set():
                        self.answer_stop_at=cutoff
                        self.answer_frames=deque((f for f in self.answer_frames if f.timestamp<=cutoff),maxlen=1000)
                        self.answer["status"]="finishing";self.answer_finish_event.set()
                if self.answer_frames:
                    elapsed=self.answer_frames[-1].timestamp-self.answer_frames[0].timestamp
                    with self.lock:self.answer.update(elapsed_seconds=elapsed,source_frames=len(self.answer_frames))
                    if (elapsed>=30 or len(self.answer_frames)>=900) and not self.answer_finish_event.is_set():
                        self.answer_stop_at=self.answer_frames[-1].timestamp
                        self.answer["status"]="finishing";self.answer_finish_event.set()
                if self.answer_finish_event.is_set():break
                source_stop.wait(.015)
            if source_stop.is_set():
                with self.lock:
                    self.answer_frames.clear();self.audio.chunks.clear()
                return
            window=select_answer(tuple(self.answer_frames),identifier=1)
            # Settle callback lag while streams are still running, never insert zeros for gaps.
            deadline=time.monotonic()+.4
            waveform,meta=self.audio.slice(window.start,window.end)
            while waveform is None and time.monotonic()<deadline and not source_stop.is_set():
                _,chunks=self.capture.drain()
                for chunk in chunks:self.audio.add(chunk)
                waveform,meta=self.audio.slice(window.start,window.end)
                source_stop.wait(.02)
            self.capture.stop()
            if any(self.capture.snapshot().get(k,{}).get("status") in ("starting","running","stopping") for k in ("camera","audio")):
                raise RuntimeError("Capture device is still stopping; analysis was not started. Restart the application if it persists.")
            with self.lock:
                self.answer_frames.clear();self.audio.chunks.clear()
                if source_stop.is_set() or self.session_id!=session:return
                self.answer["status"]="processing"
                self.answer["duration_seconds"]=window.end-window.start
                self.metrics["queue_depth"]=1
            self.work.put_nowait((window,waveform,meta,time.monotonic()))
        except Exception as exc:
            if self.capture:self.capture.stop()
            with self.lock:
                if not source_stop.is_set() and self.session_id==session:
                    self.answer.update(status="error",reason=type(exc).__name__+": "+str(exc)[:300])
                    self.errors.append(self.answer["reason"])
                self.answer_frames.clear();self.audio.chunks.clear()

    def _infer_answer(self):
        from .pipeline import public,peak_rss_mb
        from .features import signal_diagnostics
        source_stop=self.stop_event;session=self.session_id
        while not source_stop.is_set():
            if self.answer.get("status")=="error":return
            try:item=self.work.get(timeout=.1);break
            except queue.Empty:continue
        else:return
        window,waveform,audio_meta,enqueued=item;started=time.monotonic()
        claim_task=None;claim_finished=False
        try:
            clip=np.stack([f.rgb for f in window.frames]);timestamps=[f.timestamp for f in window.frames]
            with self.lock:transcript=copy.deepcopy(self.transcript)
            claim_task=self._begin_claim_check(clip,window.id,source_stop,bool(self.answer.get("visual_claim_check")))
            options={"on_transcript":claim_task.on_transcript} if claim_task is not None else {}
            if window._source_frames:
                options.update(video_frames=window._source_frames)
            result=self.fast_runtime.infer(clip,waveform,transcript,window.id,timestamps=timestamps,
                duration_s=window.end-window.start,stop_event=source_stop,**options)
            if source_stop.is_set() or self.session_id!=session:return
            model_finished=time.monotonic()
            claim_check=claim_task.finish(result.get("transcript")) if claim_task is not None else {
                "status":"not_applicable","kind":"none","reason":"Contradiction checks are disabled.",
                "source_window_id":window.id,"evidence":[],"never_confirms_truth":True,"call_made":False,"sent_keyframes":0}
            claim_finished=True
            if source_stop.is_set() or self.session_id!=session:return
            finished=time.monotonic();metadata=window.metadata()
            transcript=result.get("transcript",{})
            transcript.update(start=window.start,end=window.end,scope="video_window")
            diagnostics=signal_diagnostics(clip,waveform,16000)
            with self.lock:
                if source_stop.is_set() or self.session_id!=session or self.mode=="stopped":return
                self.latest={"window":{**metadata,"audio":audio_meta},"branches":public(result["branches"]),
                    "remote_video_enabled":result.get("remote_video_enabled",True),
                    "features":public(diagnostics),"digest":public(result.get("features",{})),"transcript":public(transcript),
                    "runtime_timings":public(result.get("timings")),"locality":public(result.get("locality")),
                    "produced_at":finished,
                    "claim_check":public(claim_check),"claim_check_enabled":claim_task is not None,
                    "visual_claim_check":bool(self.answer.get("visual_claim_check")),
                    "model_processing_ms":(model_finished-started)*1000,
                    "mock":False,"calibrated_task_output":None}
                vision=result["branches"].get("vision",{})
                valid_video=(vision.get("status")=="trained_classifier" and isinstance(vision.get("elapsed_ms"),(int,float)) and vision["elapsed_ms"]>0)
                consumed=min(window.source_frame_count,int(vision.get("encoder_unique_source_frames",0))) if valid_video else 0
                metadata.update(transport_frames=vision.get("source_frames_sent") if valid_video else None,
                    native_video_unique_frames=consumed if valid_video else None,
                    effective_video_input_fps=consumed/(window.end-window.start) if valid_video else None)
                self.latest["window"].update(metadata)
                self.metrics.update(completed_windows=1,queue_depth=0,processing_stage="idle",
                    window_latency_ms=(finished-started)*1000,queue_wait_ms=(started-enqueued)*1000,
                    result_age_seconds=finished-window.end,process_peak_rss_mb=peak_rss_mb(),
                    unique_frames_per_second=consumed/max(.001,finished-self.created),
                    video_frames_processed=consumed,
                    video_service_unique_fps=consumed/(vision["elapsed_ms"]/1000) if valid_video else None,
                    inference_updates_per_second=1/max(.001,finished-self.created),
                    measurement_window_seconds=finished-self.created)
                self.answer.update(status="complete",processing_ms=(finished-started)*1000)

        except Exception as exc:
            with self.lock:
                if not source_stop.is_set() and self.session_id==session:
                    self.answer.update(status="error",reason=type(exc).__name__+": "+str(exc)[:300])
                    self.errors.append(self.answer["reason"])

        finally:
            if claim_task is not None and not claim_finished:claim_task.cancel()
