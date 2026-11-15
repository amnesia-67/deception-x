"""Resident local encoders plus bounded remote-vision window orchestration.

No downloads, capture, manager calls, classifier execution or local vision
fallback occur here. Timed-out forwards may finish, but never supply another
window's features. Callers must retain source timestamps alongside window IDs.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import threading
import time

from .asr import clear_asr_cache, prepare_asr
from .models import CONTRACT, MODEL_IDS, ModelRegistry, ModelUnavailable


class FastRuntime:
    """One persistent model set and at most one in-flight job of each kind."""

    def __init__(self, config, vision_runner=None, on_progress=None):
        self.config = copy.deepcopy(config)
        models = self.config.setdefault("models", {})
        for branch in ("dasheng", "text", "emotion"):
            models.setdefault(branch, {})["keep_loaded"] = True
        self.asr_config = models.get("asr", self.config.get("asr", {}))
        self.asr_config["keep_loaded"] = True
        self.registry = ModelRegistry(self.config)
        self.vision_runner = vision_runner
        self.on_progress = on_progress
        options = self.config.get("fast_runtime", {})
        self.deadline_seconds = float(options.get("deadline_seconds", 15))
        if not .001 <= self.deadline_seconds <= 120:
            raise ValueError("fast_runtime.deadline_seconds must be within0.001..120")
        self.emotion_enabled = bool(options.get("emotion_enabled", True))
        self.prepare_emotion = bool(options.get("prepare_emotion", False))
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sunhacks-fast")
        self._lock = threading.RLock()
        self._mps_lock = threading.RLock()
        self._inflight = {}
        self._active_prepares = 0
        self._closed = False
        self._cleanup_started = False
        self._cleanup_done = False
        self._cleanup_finished = threading.Event()

    def _progress(self, stage, status, **details):
        if self.on_progress is not None:
            try:
                self.on_progress({"stage": stage, "status": status, **details})
            except Exception:
                # UI telemetry must not break model lifecycle or inference.
                pass

    @staticmethod
    def _unavailable(branch, reason, window_id=None, status="unavailable"):
        return {"status": status, "model_id": MODEL_IDS[branch], "elapsed_ms": 0.0,
                "embedding_shape": None, "reason": reason, "input_contract": CONTRACT,
                "source_window_id": window_id}

    def _stopped(self, stop_event=None, deadline=None):
        return (self._closed or (stop_event is not None and stop_event.is_set()) or
                (deadline is not None and time.monotonic() >= deadline))

    def prepare(self):
        """Serially load weights; no forward, microphone or camera is used.

        This is not a kernel warmup. Missing remote vision is explicitly partial
        readiness even if all required local components load successfully.
        """
        started = time.monotonic()
        with self._lock:
            if self._closed:
                return {"status": "stopped", "components": {}, "kernels_warmed": False}
            if self._active_prepares or self._inflight:
                return {"status": "busy", "components": {}, "kernels_warmed": False,
                        "reason": "Preparation cannot overlap active model jobs or another preparation."}
            self._active_prepares += 1
        components = {}
        try:
            with self._mps_lock:
                if not self._stopped() and self.asr_config.get("path"):
                    self._progress("asr", "loading")
                    components["asr"] = prepare_asr(self.asr_config)
                    self._progress("asr", components["asr"]["status"], **{
                        key: components["asr"][key] for key in ("elapsed_ms", "cache_hit")
                        if key in components["asr"]})
                else:
                    components["asr"] = {"status": "unavailable", "kernels_warmed": False,
                                         "reason": "Local ASR is unconfigured or runtime stopped; manual transcript remains supported."}
                selected = ["dasheng", "text"]
                if self.emotion_enabled and self.prepare_emotion:
                    selected.append("emotion")
                for branch in selected:
                    if self._stopped():
                        components[branch] = {"status": "cancelled", "kernels_warmed": False}
                        continue
                    branch_started = time.monotonic()
                    cache_hit = branch in self.registry._loaded
                    self._progress(branch, "loading")
                    try:
                        self.registry._load(branch)
                        item = {"status": "ready", "model_id": MODEL_IDS[branch],
                                "weights_loaded": True, "kernels_warmed": False,
                                "cache_hit": cache_hit}
                    except Exception as exc:
                        item = {"status": "unavailable" if isinstance(exc, (ModelUnavailable, OSError, ImportError)) else "error",
                                "model_id": MODEL_IDS[branch], "weights_loaded": False,
                                "kernels_warmed": False,
                                "reason": type(exc).__name__ + ": " + str(exc)[:300]}
                    item["elapsed_ms"] = round((time.monotonic() - branch_started) * 1000, 3)
                    components[branch] = item
                    self._progress(branch, item["status"], elapsed_ms=item["elapsed_ms"])
            components["vision"] = {
                "status": "configured" if self.vision_runner is not None else "unavailable",
                "weights_loaded": False, "kernels_warmed": False,
                "reason": ("Remote runner configured; remote model identity and readiness require its own verified probe."
                           if self.vision_runner is not None else
                           "Verified remote raw-video encoder is not configured; no local vision fallback.")}
            local_ready = all(components.get(branch, {}).get("status") == "ready"
                              for branch in ("dasheng", "text"))
            if self.asr_config.get("path"):
                local_ready = local_ready and components.get("asr", {}).get("status") == "ready"
            return {"status": "stopped" if self._stopped() else
                    ("ready" if local_ready and self.vision_runner is not None else "partial"),
                    "components": components, "local_weights_loaded": local_ready,
                    "readiness_scope": "local_weights_only", "remote_verified": False,
                    "kernels_warmed": False,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 3)}
        finally:
            with self._lock:
                self._active_prepares -= 1
            self._cleanup_if_ready()

    def _cleanup_if_ready(self):
        with self._lock:
            if (not self._closed or self._cleanup_started or self._inflight or
                    self._active_prepares):
                return
            self._cleanup_started = True
        try:
            try:
                self.registry.close()
            finally:
                clear_asr_cache()
        finally:
            with self._lock:
                self._cleanup_done = True
                self._cleanup_finished.set()

    def _job_done(self, key, future):
        with self._lock:
            if self._inflight.get(key) is future:
                self._inflight.pop(key, None)
        self._cleanup_if_ready()

    def _submit(self, key, function):
        with self._lock:
            if self._closed:
                return None, "stopped"
            if self._active_prepares:
                return None, "preparing"
            previous = self._inflight.get(key)
            if previous is not None and not previous.done():
                return None, "busy"
            future = self._executor.submit(function)
            self._inflight[key] = future
            future.add_done_callback(lambda done: self._job_done(key, done))
            return future, "submitted"

    def _local_branch(self, branch, clip, audio, text, window_id, stop_event, deadline):
        predicate = lambda: self._stopped(stop_event, deadline)
        self._progress(branch, "running", window_id=window_id)
        if self.config.get("models", {}).get(branch, {}).get("device") == "mps":
            acquired = self._mps_lock.acquire(timeout=max(0, deadline - time.monotonic()))
            if not acquired:
                return self._unavailable(branch, "MPS work could not start before the window deadline.", window_id)
        else:
            acquired = False
        try:
            item = self.registry.run(clip, audio, text, branches=(branch,),
                                     should_stop=predicate)[branch]
            if branch == "emotion" and self._stopped(stop_event):
                return self._unavailable(
                    branch, "Session stopped before supplemental emotion finished; its result was discarded.",
                    window_id, "cancelled")
            item["source_window_id"] = window_id
            return item
        finally:
            if acquired:
                self._mps_lock.release()

    def _semantic(self, clip, audio, transcript, window_id, stop_event, deadline):
        started = time.monotonic()
        current = copy.deepcopy(transcript)
        acquired = self._mps_lock.acquire(timeout=max(0, deadline - time.monotonic()))
        if not acquired or self._stopped(stop_event, deadline):
            if acquired:
                self._mps_lock.release()
            return {"transcript": current,
                    "text": self._unavailable("text", "Semantic work stopped before starting.", window_id, "cancelled"),
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 3)}
        try:
            if audio is not None and self.asr_config.get("path"):
                self._progress("local_transcription", "running", window_id=window_id)
                current = self.registry.transcribe(audio)
                current.update(source="local_asr", scope="video_window", source_window_id=window_id)
            text = current.get("text", "") if current.get("status") in ("available", "base") else ""
            item = self._local_branch("text", clip, audio, text, window_id, stop_event, deadline)
            return {"transcript": current, "text": item,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 3)}
        finally:
            self._mps_lock.release()

    def _vision(self, clip, window_id, stop_event, deadline):
        if self._stopped(stop_event, deadline):
            return self._unavailable("vision", "Remote work stopped before starting.", window_id, "cancelled")
        started = time.monotonic()
        self._progress("remote_vision", "running", window_id=window_id)
        item = self.vision_runner(clip, window_id)
        if not isinstance(item, dict):
            raise TypeError("vision_runner must return one branch-output dictionary")
        item = dict(item)
        item["source_window_id"] = window_id
        item["runtime_elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        return item

    def infer(self, clip, audio, transcript: dict, window_id: int, stop_event=None):
        """Run independent paths under one deadline; never reuse older features.

        A returned `_emotion_future` resolves to an emotion branch item carrying
        its own source_window_id. Publish it only if that source is still current.
        A timed-out running job retains its slot until it actually finishes.
        """
        started = time.monotonic()
        deadline = started + self.deadline_seconds
        current_transcript = copy.deepcopy(transcript)
        source_transcript = copy.deepcopy(transcript)
        branches = {name: self._unavailable(name, "Branch has not run.", window_id)
                    for name in MODEL_IDS}
        branches["fusion"]["reason"] = "Fusion is executed by the caller after required embeddings arrive."
        branches["tabular"]["reason"] = "Tabular inference is executed separately only with a verified real feature contract."
        futures = {}
        jobs = {}
        specifications = {
            "vision": lambda: self._vision(clip, window_id, stop_event, deadline),
            "dasheng": lambda: self._local_branch("dasheng", clip, audio, "", window_id, stop_event, deadline),
            "semantic": lambda: self._semantic(clip, audio, source_transcript, window_id, stop_event, deadline),
        }
        if self.emotion_enabled:
            specifications["emotion"] = lambda: self._local_branch("emotion", clip, audio, "", window_id, stop_event, deadline)
        for key, function in specifications.items():
            branch = "text" if key == "semantic" else key
            if self._stopped(stop_event, deadline):
                future, status = None, "stopped"
            elif key == "vision" and self.vision_runner is None:
                future, status = None, "unconfigured"
            else:
                future, status = self._submit(key, function)
            jobs[key] = {"status": status}
            if future is not None:
                futures[key] = future
            else:
                reason = {"busy": "Previous source window is still running; its features will not be reused.",
                          "preparing": "Model weights are still being prepared; this source window was not queued.",
                          "stopped": "Runtime or source session has stopped.",
                          "unconfigured": "Verified remote raw-video encoder is unavailable; no local vision fallback."}.get(status, status)
                branches[branch] = self._unavailable(branch, reason, window_id,
                                                     "cancelled" if status == "stopped" else "unavailable")
        if not self.emotion_enabled:
            branches["emotion"] = self._unavailable("emotion", "Supplemental emotion branch is disabled.", window_id)

        required = {future: key for key, future in futures.items() if key != "emotion"}
        while required and not self._stopped(stop_event, deadline):
            done, _ = wait(required, timeout=min(.05, max(0, deadline - time.monotonic())),
                           return_when=FIRST_COMPLETED)
            for future in done:
                key = required.pop(future)
                branch = "text" if key == "semantic" else key
                try:
                    output = future.result()
                    if key == "semantic":
                        current_transcript, branches["text"] = output["transcript"], output["text"]
                        jobs[key]["elapsed_ms"] = output["elapsed_ms"]
                    else:
                        branches[branch] = output
                    jobs[key]["status"] = "complete"
                except Exception as exc:
                    branches[branch] = self._unavailable(branch, type(exc).__name__ + ": " + str(exc)[:300], window_id, "error")
                    jobs[key]["status"] = "error"
        for future, key in required.items():
            stopped = self._closed or (stop_event is not None and stop_event.is_set())
            branch = "text" if key == "semantic" else key
            future.cancel()
            jobs[key]["status"] = "stopped" if stopped else "timeout"
            branches[branch] = self._unavailable(
                branch, "Session stopped before this source window finished." if stopped else
                "Shared window deadline exceeded; late features are discarded.", window_id,
                "cancelled" if stopped else "unavailable")

        result = {"branches": branches, "transcript": current_transcript,
                  "timings": {"service_ms": round((time.monotonic() - started) * 1000, 3),
                              "deadline_ms": self.deadline_seconds * 1000, "jobs": jobs}}
        emotion = futures.get("emotion")
        if emotion is not None:
            stopped = self._closed or (stop_event is not None and stop_event.is_set())
            if stopped:
                emotion.cancel()
                branches["emotion"] = self._unavailable(
                    "emotion", "Session stopped; supplemental emotion will not be published.",
                    window_id, "cancelled")
                jobs["emotion"]["status"] = "stopped"
            elif emotion.done():
                try:
                    branches["emotion"] = emotion.result()
                    jobs["emotion"]["status"] = "complete"
                except Exception as exc:
                    branches["emotion"] = self._unavailable("emotion", type(exc).__name__ + ": " + str(exc)[:300], window_id, "error")
                    jobs["emotion"]["status"] = "error"
            else:
                branches["emotion"] = self._unavailable("emotion", "Supplemental emotion is running independently and does not block fusion.", window_id)
                branches["emotion"]["background_pending"] = True
                jobs["emotion"]["status"] = "background"
                result["_emotion_future"] = emotion
        result["timings"]["service_ms"] = round((time.monotonic() - started) * 1000, 3)
        return result

    def snapshot(self):
        """Read-only lifecycle state; contains no model features or media."""
        with self._lock:
            return {"closed": self._closed, "cleanup_done": self._cleanup_done,
                    "cleanup_in_progress": self._cleanup_started and not self._cleanup_done,
                    "active_jobs": sorted(self._inflight),
                    "active_prepares": self._active_prepares}

    def close(self):
        """Reject new work and cancel queued jobs; defer cleanup until active jobs end."""
        with self._lock:
            self._closed = True
            for future in tuple(self._inflight.values()):
                future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._cleanup_if_ready()
        with self._lock:
            return {"status": "closed" if self._cleanup_done else "closing",
                    "active_jobs": len(self._inflight), "active_prepares": self._active_prepares,
                    "cleanup_in_progress": self._cleanup_started and not self._cleanup_done}
