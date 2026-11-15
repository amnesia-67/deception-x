"""Ephemeral guided practice and server-timed seven-second showcase answers.

Warmup prompts are practice, not training or personal calibration. Their truth
labels are instructions to the participant; this module does not verify them.
Raw warmup images are discarded on drain and PCM stays in a two-second buffer.
"""
from __future__ import annotations

from collections import deque
import copy
import math
import threading
import time

import numpy as np

from .temporal import AudioBuffer

PROMPTS = (
    {"index": 0, "title": "First, a truth", "body": "Say your name or nickname.", "label": "truth"},
    {"index": 1, "title": "One more truth", "body": "Say your school, role, or something true about your day.", "label": "truth"},
    {"index": 2, "title": "Now, a harmless lie", "body": "Make up one harmless statement about yourself.", "label": "lie"},
)
ACTIVE = frozenset(("starting", "warmup", "warmup_review", "ready", "answer_starting", "recording", "processing"))
SENSOR_ACTIVE = frozenset(("starting", "running", "stopping"))


def _running(capture):
    return any(capture.get(key, {}).get("status") in SENSOR_ACTIVE for key in ("camera", "audio"))


def _default_capture(**kwargs):
    from .capture import LiveCapture
    return LiveCapture(**kwargs)


class ShowcaseSession:
    def __init__(self, pipeline, *, capture_factory=None, clock=None, wait=None, spawn=True):
        self.pipeline = pipeline
        self._capture_factory = capture_factory or _default_capture
        self._clock = clock or time.monotonic
        self._wait = wait or (lambda event, duration: event.wait(duration))
        self._spawn = bool(spawn)
        self._lock = threading.RLock()
        self._operations = threading.RLock()
        self._generation = 0
        self._event = threading.Event()
        self._thread = None
        self._warmup_capture = None
        self._answer_session = None
        self._warmup_done = False
        self._closed = False
        # Operator setting from the dashboard; the public showcase page has no control for it.
        self._claim_check_enabled = True
        self._state = self._fresh_state()

    @staticmethod
    def _fresh_state():
        return {"stage": "idle", "elapsed_seconds": 0.0, "remaining_seconds": 15.0,
                "prompt": dict(PROMPTS[0]), "result": None, "error": None,
                "warmup": {"completed": False, "personal_calibration": False,
                           "content_verified": False, "purpose": "guided_practice_and_signal_check"}}

    def _pipeline_state(self):
        return self.pipeline.state()

    def _current(self, generation, event):
        with self._lock:
            return not self._closed and self._generation == generation and self._event is event and not event.is_set()

    def _publish(self, generation, event, **values):
        with self._lock:
            if self._current(generation, event):
                self._state.update(values)
                return True
        return False

    def _capture_status(self):
        with self._lock:
            capture = self._warmup_capture
        return capture.snapshot() if capture is not None else None

    def is_active(self):
        with self._lock:
            active = self._state["stage"] in ACTIVE
            thread = self._thread
        capture = self._capture_status()
        return bool(active or (thread is not None and thread.is_alive()) or (capture and _running(capture)))

    def _preparing(self, runtime):
        thread = getattr(self.pipeline, "prepare_thread", None)
        thread_alive = bool(thread is not None and thread.is_alive())
        return bool(runtime.get("status") == "preparing" or thread_alive or
                    runtime.get("lifecycle", {}).get("active_prepares"))

    def snapshot(self):
        with self._lock:
            value = copy.deepcopy(self._state)
            done = self._warmup_done
            closed = self._closed
            claim_check_enabled = self._claim_check_enabled
        pipeline = self._pipeline_state()
        runtime = pipeline.get("runtime", {})
        runtime_blocked=runtime.get("status")=="blocked"
        model_ready = bool(runtime.get("live_ready")) and not closed and not runtime_blocked
        preparing = self._preparing(runtime) and not closed
        lifecycle = runtime.get("lifecycle", {})
        pipeline_idle = (pipeline.get("answer", {}).get("status") not in ("capturing", "finishing", "processing")
                         and not _running(pipeline.get("capture", {})) and pipeline.get("mode") != "demo" and not (pipeline.get("dataset") or {}).get("active"))
        can_prepare = (not runtime_blocked and not closed and not preparing and not self.is_active() and pipeline_idle
                       and not lifecycle.get("active_jobs") and not lifecycle.get("active_prepares"))
        warmup_capture = self._capture_status()
        use_warmup = value["stage"] in ("starting", "warmup", "warmup_review") or bool(warmup_capture and _running(warmup_capture))
        value.update(model_ready=model_ready, preparing=preparing, can_prepare=bool(can_prepare),runtime_blocked=runtime_blocked,
                     remote_video_enabled=runtime.get("remote_video_enabled",True),
                     claim_check_enabled=claim_check_enabled,
                     can_toggle_remote_video=bool(not closed and not preparing and not self.is_active() and pipeline_idle
                         and not lifecycle.get("active_jobs") and not lifecycle.get("active_prepares")),
                     ready=(model_ready and pipeline_idle and not lifecycle.get("active_jobs")
                            and not lifecycle.get("active_prepares")
                            and value["stage"] in ("idle", "ready", "result")),
                     model_reason=runtime.get("reason", "Prepare local models before starting."),
                     readiness_reason=runtime.get("reason", "Prepare local models before starting."),
                     capture=warmup_capture if use_warmup else pipeline.get("capture", {}),
                     locality=runtime.get("locality",{"video": "friend_gpu", "audio": "mac", "transcript": "mac", "ensemble": "mac"}),
                     privacy="GPU video is optional. Warmup stays in memory; the transcript/photo contradiction check has its own setting.")
        return value

    def _assert_idle_pipeline(self, state):
        if (state.get("dataset") or {}).get("active"):raise RuntimeError("Label or discard the dataset recording before starting the showcase.")
        if state.get("answer", {}).get("status") in ("capturing", "finishing", "processing") or _running(state.get("capture", {})):
            raise RuntimeError("Stop the active recording before starting the showcase.")
        lifecycle = state.get("runtime", {}).get("lifecycle", {})
        if lifecycle.get("active_jobs") or lifecycle.get("active_prepares"):
            raise RuntimeError("Previous model work is still finishing.")
        if state.get("mode") == "demo":
            raise RuntimeError("Stop the engineering demo before starting the showcase.")

    def _begin(self, stage):
        with self._lock:
            if self._closed:
                raise RuntimeError("The showcase is closed; restart the application.")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("The previous showcase action is still finishing.")
            if self._warmup_capture is not None and _running(self._warmup_capture.snapshot()):
                raise RuntimeError("The capture device is still stopping; restart the application if it persists.")
            self._generation += 1
            self._event = threading.Event()
            self._state.update(stage=stage, elapsed_seconds=0.0, result=None, error=None)
            return self._generation, self._event

    def _launch(self, target, generation, event, name):
        if self._spawn:
            thread = threading.Thread(target=target, args=(generation, event), name=name, daemon=True)
            with self._lock:
                self._thread = thread
                thread.start()

    def prepare_models(self):
        """Recheck/reconnect resident models without opening capture devices."""
        with self._operations:
            with self._lock:
                if self._closed:
                    raise RuntimeError("The showcase is closed; restart the application.")
            if self.is_active():
                raise RuntimeError("Cancel the active showcase before checking model connections.")
            state = self._pipeline_state()
            self._assert_idle_pipeline(state)
            if self._preparing(state.get("runtime", {})):
                raise RuntimeError("Model preparation is already running.")
            self.pipeline.prepare_runtime()
        return self.snapshot()

    def configure_video(self, enabled):
        with self._operations:
            if self._closed:raise RuntimeError("The showcase is closed")
            if self.is_active():raise RuntimeError("Cancel or reset the current test before changing GPU video")
            state=self._pipeline_state()
            self._assert_idle_pipeline(state)
            if self._preparing(state.get("runtime",{})):raise RuntimeError("Wait for model preparation to finish")
            self.pipeline.set_remote_video(enabled)
            self.pipeline.clear_session_data()
            with self._lock:
                self._warmup_done=False;self._answer_session=None;self._state=self._fresh_state()
        return self.snapshot()

    def claim_check_enabled(self):
        with self._lock:
            return self._claim_check_enabled

    def configure_claim_check(self, enabled):
        """Applies to the next answer; an answer already started keeps its own setting."""
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        with self._lock:
            if self._closed:raise RuntimeError("The showcase is closed")
            self._claim_check_enabled = enabled
        return self.snapshot()

    def start_warmup(self):
        with self._operations:
            if self.is_active():
                raise RuntimeError("The showcase is active. Cancel or reset before a new warmup.")
            state = self._pipeline_state()
            self._assert_idle_pipeline(state)
            if not state.get("runtime", {}).get("live_ready"):
                raise RuntimeError(state.get("runtime", {}).get("reason", "Models are not ready."))
            clear = getattr(self.pipeline, "clear_session_data", None)
            if callable(clear):
                clear()
            generation, event = self._begin("starting")
            with self._lock:
                self._warmup_done = False
                self._answer_session = None
                self._state = {**self._fresh_state(), "stage": "starting"}
            self._launch(self._run_warmup, generation, event, "showcase-warmup")
        return self.snapshot()

    def _run_warmup(self, generation, event):
        capture = None
        audio = AudioBuffer(retention_seconds=2.0)
        candidates = deque(maxlen=120)
        started = self._clock()
        timer = None
        last_frame = last_audio = None
        sample_count = clipping = frame_count = 0
        square_sum = 0.0
        complete = False
        failure = None
        try:
            if not self._current(generation, event):
                return
            cfg = self.pipeline.config
            capture = self._capture_factory(camera_index=cfg.get("camera_device", 0),
                       audio_device=cfg.get("audio_device"), camera_fps=cfg.get("capture_fps", 30),
                       max_video_frames=30, max_audio_chunks=30)
            with self._lock:
                if not self._current(generation, event):
                    return
                self._warmup_capture = capture
                capture.start()
            while self._current(generation, event):
                now = self._clock()
                frames, chunks = capture.drain()
                for chunk in chunks:
                    audio.add(chunk)
                    last_audio = chunk.start + len(chunk.samples) / chunk.sample_rate
                    if timer is not None:
                        begin = max(0, round((timer - chunk.start) * chunk.sample_rate))
                        end = min(len(chunk.samples), round((timer + 15.0 - chunk.start) * chunk.sample_rate))
                        values = chunk.samples[begin:max(begin, end)]
                        square_sum += float(np.dot(values.astype(np.float64), values.astype(np.float64)))
                        sample_count += int(values.size)
                        clipping += int(np.count_nonzero(np.abs(values) >= .999))
                for frame in frames:
                    candidates.append((frame.timestamp, frame.sequence))
                    last_frame = frame.timestamp
                    frame_count += 1
                # Drop image references immediately; no warmup video is retained.
                frames.clear()
                statuses = capture.snapshot()
                for key in ("camera", "audio"):
                    status = statuses.get(key, {})
                    if status.get("status") in ("missing", "error", "denied"):
                        raise RuntimeError(key + ": " + str(status.get("error", "Capture unavailable")))
                if statuses.get("audio", {}).get("overflow_callbacks", 0) or statuses.get("audio", {}).get("resampling_discontinuities", 0):
                    raise RuntimeError("Microphone input was interrupted. Check the microphone and try again.")
                if timer is None and all(statuses.get(k, {}).get("status") == "running" for k in ("camera", "audio")):
                    for stamp, _ in candidates:
                        observed, _ = audio.slice(stamp, stamp + .001)
                        if observed is not None:
                            timer = stamp
                            break
                if timer is None:
                    if now - started > 8:
                        raise RuntimeError("Camera and microphone did not become ready within eight seconds.")
                else:
                    if last_frame is None or now - last_frame > 2:
                        raise RuntimeError("Camera stopped delivering frames during warmup.")
                    if last_audio is None or now - last_audio > 2:
                        raise RuntimeError("Microphone stopped delivering audio during warmup.")
                    elapsed = max(0.0, min(15.0, now - timer))
                    index = min(2, int(elapsed // 5))
                    self._publish(generation, event, stage="warmup", elapsed_seconds=elapsed,
                                  remaining_seconds=max(0.0, 15.0-elapsed), prompt=dict(PROMPTS[index]))
                    if now >= timer + 15.0:
                        complete = True
                        self._publish(generation, event, stage="warmup_review", elapsed_seconds=15.0, remaining_seconds=0.0)
                        break
                self._wait(event, .02)
            if complete and (not sample_count or not math.isfinite(square_sum) or square_sum <= sample_count * 1e-10):
                raise RuntimeError("No microphone signal was observed. Check your input and try the warmup again.")
        except Exception as exc:
            failure = type(exc).__name__ + ": " + str(exc)[:350]
        finally:
            if capture is not None:
                try:
                    capture.stop()
                    if _running(capture.snapshot()):
                        failure = "Capture is still stopping. Restart the application before another recording."
                except Exception as exc:
                    failure = "Capture cleanup failed: " + type(exc).__name__
            audio.chunks.clear()
            candidates.clear()
            if self._current(generation, event):
                if failure:
                    self._publish(generation, event, stage="error", error=failure)
                elif complete:
                    mean_square = square_sum / sample_count
                    warmup = {"completed": True, "personal_calibration": False, "content_verified": False,
                              "purpose": "guided_practice_and_signal_check", "duration_seconds": 15.0,
                              "observed_audio_seconds": sample_count / 16000,
                              "rms_dbfs": 10.0 * math.log10(max(1e-12, mean_square)),
                              "clipping_fraction": clipping / sample_count, "source_frames": frame_count,
                              "signal_present": True}
                    with self._lock:
                        if self._current(generation, event):
                            self._warmup_done = True
                            self._state.update(stage="ready", warmup=warmup, remaining_seconds=7.0,
                                               elapsed_seconds=0.0, prompt=None)

    def start_answer(self, *, visual_claim_check=False):
        if type(visual_claim_check) is not bool:
            raise ValueError("Image authorization must be an explicit boolean")
        with self._operations:
            with self._lock:
                previous_stage = self._state["stage"]
                if previous_stage not in ("idle", "cancelled", "error", "ready", "result"):
                    raise RuntimeError("Finish or cancel the active showcase before starting another answer.")
            state = self._pipeline_state()
            self._assert_idle_pipeline(state)
            if not state.get("runtime", {}).get("live_ready"):
                raise RuntimeError(state.get("runtime", {}).get("reason", "Models are not ready."))
            if previous_stage in ("idle", "cancelled", "error"):
                clear = getattr(self.pipeline, "clear_session_data", None)
                if callable(clear):
                    clear()
            generation, event = self._begin("answer_starting")
            self._publish(generation, event, remaining_seconds=7.0, prompt=None,visual_claim_check=visual_claim_check)
            try:
                options={"auto_stop_seconds":7.0}
                if visual_claim_check:options["visual_claim_check"]=True
                state = self.pipeline.start("live", **options)
                with self._lock:
                    self._answer_session = state.get("session_id")
                if not self._answer_session:
                    raise RuntimeError("The answer did not receive a source session.")
            except Exception as exc:
                self._publish(generation, event, stage="error", error=type(exc).__name__ + ": " + str(exc)[:350])
                raise
            self._launch(self._monitor_answer, generation, event, "showcase-answer")
        return self.snapshot()

    def _monitor_answer(self, generation, event):
        started = self._clock()
        while self._current(generation, event):
            state = self._pipeline_state()
            with self._lock:
                source_session = self._answer_session
            if state.get("session_id") != source_session:
                self._publish(generation, event, stage="error", error="The answer session changed; the old result was discarded.")
                return
            answer = state.get("answer", {})
            status = answer.get("status")
            elapsed = max(0.0, min(7.0, float(answer.get("elapsed_seconds", 0.0))))
            if status == "complete":
                result = self._result(state, source_session)
                with self._lock:
                    result["practice_completed"] = self._warmup_done
                self._publish(generation, event, stage="result", result=result, remaining_seconds=0.0,
                              elapsed_seconds=elapsed)
                return
            if status in ("error", "stopped"):
                self._publish(generation, event, stage="error" if status == "error" else "cancelled",
                              error=answer.get("reason", "The answer was cancelled."))
                return
            if status == "capturing":
                stage = "recording" if answer.get("timer_started_at") is not None else "answer_starting"
            elif status in ("finishing", "processing"):
                stage = "processing"
            else:
                stage = "answer_starting"
            self._publish(generation, event, stage=stage, elapsed_seconds=elapsed,
                          remaining_seconds=max(0.0, 7.0-elapsed))
            # Independent failsafe; the capture path owns the actual seven-second deadline.
            if self._clock() - started > 45:
                if self._current(generation, event) and self.pipeline.session_id == source_session:
                    self.pipeline.stop()
                self._publish(generation, event, stage="error", error="The answer exceeded its capture/inference deadline.")
                return
            self._wait(event, .05)

    @staticmethod
    def _finite_number(value):
        return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)

    @staticmethod
    def _quality_reason(latest):
        features=latest.get("features",{})
        audio=features.get("audio",{}) if isinstance(features,dict) else {}
        video=features.get("video",{}) if isinstance(features,dict) else {}
        numeric=ShowcaseSession._finite_number
        if audio.get("status")!="ok" or any(not numeric(audio.get(key)) for key in
                ("rms_dbfs","peak_dbfs","clipping_fraction","activity_fraction_proxy","duration_seconds")):
            return "Microphone quality diagnostics are missing or invalid; re-record this answer."
        if (not 0<=audio["clipping_fraction"]<=1 or not 0<=audio["activity_fraction_proxy"]<=1
                or audio["duration_seconds"]<=0 or audio.get("sample_rate")!=16000):
            return "Microphone quality diagnostics do not match the capture contract."
        # -160dB is the explicit numerical floor in audio_diagnostics, not a
        # learned speech/VAD threshold or a claim about deception confidence.
        if audio["rms_dbfs"]<=-160 and audio["peak_dbfs"]<=-160:
            return "No usable microphone signal was observed; a transcript alone is not speech evidence."
        if audio["clipping_fraction"]==1.0:
            return "Every microphone sample is clipped; reduce the input level and re-record."
        if video.get("status")!="ok" or any(not numeric(video.get(key)) for key in
                ("brightness_mean","mean_absolute_frame_difference")):
            return "Camera quality diagnostics are missing or invalid; re-record this answer."
        if (not 0<=video["brightness_mean"]<=1 or not 0<=video["mean_absolute_frame_difference"]<=1
                or type(video.get("frames")) is not int or video["frames"]!=128):
            return "Camera quality diagnostics do not match the captured answer."
        digest=latest.get("digest",{})
        face=digest.get("face",{}) if isinstance(digest,dict) else {}
        rate=face.get("face_detect_rate")
        if face.get("status")!="available" or not numeric(rate) or not 0<=rate<=1 or not isinstance(face.get("centre_crop_fallback"),bool):
            return "Participant-face diagnostics are missing or invalid; re-record facing the camera."
        if rate==0 or face["centre_crop_fallback"]:
            return "No participant face was detected; re-record with your face visible to the camera."
        return None

    @staticmethod
    def _result(state, source_session):
        answer = state.get("answer", {})
        latest = state.get("latest") or {}
        video_enabled=latest.get("remote_video_enabled",True)
        branches = latest.get("branches", {})
        window = latest.get("window", {})
        transcript = latest.get("transcript", {})
        aliases={"ensemble":"stack","dasheng":"dasheng","digest":"tabular","video":"vision"}
        scores={name:(branches.get(key,{}).get("raw_score") if ShowcaseSession._finite_number(branches.get(key,{}).get("raw_score")) else None)
                for name,key in aliases.items()}
        fold_ranges={}
        for name,key in (("dasheng","dasheng"),("video","vision")):
            values=branches.get(key,{}).get("per_fold_scores")
            if isinstance(values,(list,tuple)) and values and all(ShowcaseSession._finite_number(x) and 0<=x<=1 for x in values):
                fold_ranges[name]={"min":min(values),"max":max(values),"count":len(values)}
        stack=branches.get("stack",{})
        pilot=stack.get("profile")=="webcam_pilot"
        manual=pilot and stack.get("manual_blend") is True
        video_used=manual and stack.get("weights",{}).get("vision",0)>0
        threshold=stack.get("threshold")
        threshold_valid=ShowcaseSession._finite_number(threshold) and 0<=threshold<=1
        provenance=stack.get("threshold_provenance")
        common = {"label": "inconclusive", "score": None,
                  "threshold":float(threshold) if threshold_valid else None,
                  "threshold_provenance":provenance if isinstance(provenance,str) else None,"calibrated": False,
                  "model_result_valid":False,"model_label":None,
                  "branch_scores":{name:None for name in aliases},"distance_from_threshold":None,
                  "video_used_in_verdict":video_used,"fold_score_ranges":{},
                  "manual_blend":manual,"weights":copy.deepcopy(stack.get("weights",{})),
                  "remote_video_enabled":video_enabled,
                  "classifier_profile":"webcam_pilot" if pilot else "courtroom",
                  "validation_context":{"source":"unvalidated manual pilot blend" if manual else "webcam-pilot audio export" if pilot else "deployed ensemble export","model_id":stack.get("model_id"),
                      "calibration_scope":stack.get("calibration_scope"),
                      "reported_cv_metrics":copy.deepcopy(stack.get("cv_metrics")),
                      "caveats":copy.deepcopy(stack.get("caveats",[]))},
                  "processing_ms": answer.get("processing_ms"), "duration_seconds": window.get("duration_seconds"),
                  "source_session_id": source_session,
                  "source_window_id": window.get("id"), "video_in_local_ensemble": video_used,
                  "decision_rule": "Weighted calibrated branch margins determine the experimental verdict." if manual else "Class-1 score at or above the reviewed export's threshold means deceptive; not a personal cutoff.",
                  "uncertainty": "Experimental prediction; not evidence that a person is lying."}
        required = {"vision": ("trained_classifier",), "dasheng": ("trained", "trained_classifier"),
                    "text": ("base",), "emotion": ("base",),
                    "tabular": ("trained_classifier",), "stack": ("trained_classifier",)}
        if pilot:
            for name in ("text","emotion","tabular"):required.pop(name)
        if manual:
            required["stack"]=("experimental_blend",)
            if video_used and not video_enabled:
                return {**common,"reason":"This blend requires a same-answer GPU video result."}
        if not video_enabled:
            required.pop("vision");scores["video"]=None;fold_ranges.pop("video",None)
        if (state.get("session_id") != source_session or latest.get("mock") is not False or
                not window.get("id") or window.get("capture_mode") != "whole_answer"):
            return {**common, "reason": "A matching real answer result is unavailable."}
        duration = window.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 5 <= duration <= 7.2:
            return {**common, "reason": "The answer did not match the seven-second capture contract."}
        for name, statuses in required.items():
            branch = branches.get(name, {})
            if branch.get("status") not in statuses or branch.get("source_window_id") != window["id"]:
                return {**common, "reason": "The " + name + " model did not complete for this answer."}
        audio = window.get("audio", {})
        if (audio.get("status") != "available" or audio.get("coverage") != 1.0 or
                audio.get("missing_samples") != 0 or transcript.get("source") != "local_asr" or
                not isinstance(transcript.get("text"), str) or not transcript["text"].strip() or
                transcript.get("scope") != "video_window"):
            return {**common, "reason": "Complete audio and a recognized same-answer transcript are required."}
        for name,key in aliases.items():
            if (name=="video" and not video_enabled) or (pilot and name=="digest"):continue
            value=branches[key].get("raw_score")
            if not ShowcaseSession._finite_number(value) or not 0<=value<=1:
                return {**common,"reason":"The "+name+" classifier did not return a valid score."}
        if not threshold_valid:
            return {**common,"reason":"The reviewed classifier decision threshold is missing or invalid."}
        quality=ShowcaseSession._quality_reason(latest)
        if quality:return {**common,"reason":quality}
        score = branches["stack"]["raw_score"]
        common.update(branch_scores=scores,distance_from_threshold=abs(score-threshold),fold_score_ranges=fold_ranges)
        value={**common, "label":"lie" if score>=threshold else "truth", "model_label":"lie" if score>=threshold else "truth",
               "model_result_valid":True,"score":float(score),"recognized_text":transcript["text"],
               "reason":"Prediction from the user-selected experimental blend; accuracy has not been validated." if manual else ("Prediction from the selected webcam-pilot audio model; GPU video is " if pilot else "Prediction from the trained local Dasheng and behavioral ensemble; GPU video is ")+("checked separately." if video_enabled else "disabled.")}
        return ShowcaseSession._guarded_verdict(value,latest)

    @staticmethod
    def _guarded_verdict(value,latest):
        claim=latest.get("claim_check")
        if claim is None and not latest.get("claim_check_enabled"):
            value["claim_check"]={"status":"not_applicable","kind":"none","reason":"No contradiction check was requested.","evidence":[]}
            return value
        source=value.get("source_window_id")
        if (not isinstance(claim,dict) or type(claim.get("source_window_id")) is not int or claim["source_window_id"]!=source
                or claim.get("status") not in ("contradiction","no_contradiction","unverifiable","not_applicable","cancelled")):
            value.update(label="inconclusive",reason="A matching contradiction-check result is unavailable.",
                         claim_check={"status":"unverifiable","kind":"none","reason":"Mismatched or missing guard output was discarded.","evidence":[]})
            return value
        value["claim_check"]=copy.deepcopy(claim)
        if claim["status"]=="contradiction":
            value.update(label="lie",decision_source="supported_contradiction",reason=claim.get("reason","The transcribed claim contradicts the available evidence."))
        elif claim["status"]=="cancelled":
            value.update(label="inconclusive",reason="The source contradiction check was cancelled.")
        return value

    def cancel(self):
        with self._operations:
            with self._lock:
                old_event = self._event
                old_event.set()
                self._generation += 1
                source_session = self._answer_session
                capture = self._warmup_capture
                thread = self._thread
                self._state.update(stage="cancelled", result=None, error=None)
            if capture is not None:
                capture.stop()
            if source_session is not None and self.pipeline.session_id == source_session:
                self.pipeline.stop()
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.5)
            if capture is not None and _running(capture.snapshot()):
                with self._lock:
                    self._state.update(stage="error", error="Capture is still stopping. Restart the application.")
        return self.snapshot()

    def reset(self):
        self.cancel()
        with self._operations:
            if self.is_active():
                raise RuntimeError("Capture or model work is still stopping.")
            clear = getattr(self.pipeline, "clear_session_data", None)
            if callable(clear):
                clear()
            with self._lock:
                self._warmup_done = False
                self._answer_session = None
                self._state = self._fresh_state()
        return self.snapshot()

    def close(self):
        with self._operations:
            with self._lock:
                self._closed = True
            self.cancel()
        return {"status": "closed"}
