"""Bounded, answer-scoped dispatch for optional contradiction checks.

Only the owned workers call the checker. No model or network work occurs in the
transcript callback. A checker must cooperate with its stop event and deadline;
Python cannot forcibly stop an executing thread. Its slot remains occupied until
it actually exits, even when the answer has already timed out or been cancelled.
"""
from __future__ import annotations

import copy
import math
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor


_STATUSES = frozenset(("contradiction", "no_contradiction", "not_applicable", "unverifiable", "cancelled"))
_KINDS = frozenset(("none", "math", "visual", "multimodal"))
_SCOPES = frozenset(("video_window", "answer", "same_answer"))
_MAX_TEXT = 4000
_MAX_IMAGE_CHARS = ((2 * 1024 * 1024 + 2) // 3) * 4


def _source_id(value):
    return type(value) is int and 0 <= value <= 2**53 - 1


def _event_set(event):
    if event is None:
        return False
    try:
        return bool(event.is_set())
    except Exception:
        # An unusable cancellation source must never permit a remote request.
        return True


class _ScopedStop:
    def __init__(self, source):
        self.source = source
        self.owned = threading.Event()

    def is_set(self):
        return self.owned.is_set() or _event_set(self.source)


class ClaimRuntime:
    """Own at most two answer tasks and two executing checker calls, with no queue.

    ``config`` is the ``claim_check`` configuration subsection. The default
    ClaimChecker is imported and constructed lazily inside the first worker.
    An injected checker is the lightweight test/integration seam.
    """

    def __init__(self, config=None, checker=None):
        self.config = dict(config or {})
        self._checker = checker
        value = getattr(checker, "deadline", self.config.get("deadline_seconds", 5.0))
        # Injected fakes may use shorter deadlines than the real checker's
        # 0.1-second minimum so lifecycle tests need no long waits.
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        if not valid or value <= 0 or value > 15 or (checker is None and value < .1):
            raise ValueError("A finite bounded checker deadline is required")
        self.deadline = float(value)
        self._lock = threading.RLock()
        self._checker_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="answer-claim")
        self._tasks = weakref.WeakSet()
        self._reserved = weakref.WeakSet()
        self._closed = False

    def begin(self, images=(), visual_authorized=False, *, source_window_id, stop_event=None):
        if not _source_id(source_window_id):
            raise ValueError("A valid integer source window is required")
        task = AnswerClaimTask(self, source_window_id, stop_event)
        with self._lock:
            self._tasks.add(task)
            if self._closed:
                task._cancel_requested = True
                task._stop.owned.set()
                task._result = task._common("cancelled", "The claim runtime is closed.")
                return task
            if len(self._reserved) >= 2:
                task._result = task._common("unverifiable", "The two claim-check worker slots are occupied; no check was queued.")
                return task
            self._reserved.add(task)
        if type(visual_authorized) is not bool:
            task._reject("Selected-frame authorization must be an explicit boolean.")
        elif not visual_authorized:
            # Do not iterate, copy, retain or pass images without authorization.
            task._visual_authorized = False
            task._images = ()
        elif not isinstance(images, (tuple, list)) or len(images) not in (0, 2) or any(not isinstance(value, str) or len(value) > _MAX_IMAGE_CHARS for value in images):
            task._reject("Selected frames do not satisfy the bounded image contract.")
        else:
            task._visual_authorized = True
            task._images = tuple(images)
        return task

    def _get_checker(self):
        with self._checker_lock:
            if self._checker is None:
                from .claim_check import ClaimChecker
                self._checker = ClaimChecker(self.config)
            return self._checker

    def _submit(self, task, text):
        with self._lock:
            if self._closed or task._stop.is_set():
                return None
            if task not in self._reserved:
                return None
            # Capture the immutable image tuple for this answer only. No ambient
            # current-window/session state is read by the executing worker.
            images = task._images
            authorized = task._visual_authorized
            def check():
                try:
                    checker = self._get_checker()
                    if task._stop.is_set():
                        return task._common("cancelled", "The source answer was cancelled before checking.")
                    if time.monotonic() >= task._deadline_at:
                        return task._common("unverifiable", "The claim-check deadline expired before checking.")
                    task._started.set()
                    return checker.check(text, images=images, visual_authorized=authorized,
                                         source_window_id=task.source_window_id, stop_event=task._stop)
                finally:
                    # Record before Future becomes done so finish cannot accept
                    # a late response just because it was collected afterwards.
                    task._completed_at = time.monotonic()
            task._submitted_at = time.monotonic()
            task._deadline_at = task._submitted_at + self.deadline + .2
            future = self._executor.submit(check)
            task._future = future
            future.add_done_callback(lambda _: self._released(task))
            return future

    def _released(self, task):
        # This callback runs only when the real worker has exited, never when a
        # caller merely stops waiting for it.
        with task._lock:
            task._images = ()
        with self._lock:
            self._reserved.discard(task)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            tasks = list(self._tasks)
        for task in tasks:
            task._cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)


class AnswerClaimTask:
    """One callback attempt and one fail-closed result for one captured answer."""

    def __init__(self, runtime, source_window_id, stop_event):
        self._runtime = runtime
        self.source_window_id = source_window_id
        self._stop = _ScopedStop(stop_event)
        self._lock = threading.RLock()
        self._started = threading.Event()
        self._future = None
        self._attempted = False
        self._submitted_at = None
        self._deadline_at = None
        self._completed_at = None
        self._cancel_requested = False
        self._images = ()
        self._visual_authorized = False
        self._result = None

    def _common(self, status, reason, *, unknown=False):
        value = {"source_window_id": self.source_window_id, "status": status, "kind": "none",
                 "reason": reason, "evidence": [], "call_made": None if unknown else False,
                 "sent_keyframes": None if unknown else 0, "sent_transcript":None if unknown else False, "never_confirms_truth": True,
                 "decision_effect": "contradiction_only"}
        if unknown:
            value.update(call_outcome_unknown=True, attempted_keyframes=2, attempted_transcript=True,
                         transmission_status="worker_outcome_unknown")
        return value

    def _unknown(self):
        return self._started.is_set() and self._visual_authorized and self._had_images

    @property
    def _had_images(self):
        # Recorded at dispatch because the owned worker clears retained images.
        return getattr(self, "_dispatched_image_count", 0) == 2

    def _reject(self, reason):
        with self._lock:
            self._result = self._common("unverifiable", reason)
            self._images = ()
        self._runtime._released(self)

    def cancel(self):
        """Cancel this owned answer task without publishing its evidence."""
        self._cancel()

    def _cancel(self):
        with self._lock:
            self._cancel_requested = True
            self._stop.owned.set()
            future = self._future
            self._images = ()
        if future is not None:
            future.cancel()
        else:
            self._runtime._released(self)

    def _cancelled(self):
        return self._cancel_requested or _event_set(self._stop.source)

    def on_transcript(self, value):
        """Validate and submit once; this callback never waits for a worker."""
        with self._lock:
            if self._attempted or self._result is not None:
                return
            self._attempted = True
            if self._cancelled() or self._stop.is_set():
                self._cancel()
                return
            valid = (isinstance(value, dict) and _source_id(value.get("source_window_id"))
                     and value["source_window_id"] == self.source_window_id
                     and value.get("source") == "local_asr" and isinstance(value.get("scope"), str) and value["scope"] in _SCOPES
                     and isinstance(value.get("status"), str) and value["status"] in ("base", "available")
                     and value.get("truncated", False) is False
                     and isinstance(value.get("text"), str) and bool(value["text"].strip())
                     and len(value["text"]) <= _MAX_TEXT)
            if not valid:
                self._reject("A complete, same-answer local ASR transcript is required; no claim check was dispatched.")
                return
            self._dispatched_image_count = len(self._images)
            try:
                future = self._runtime._submit(self, value["text"])
            except Exception:
                self._reject("The claim check could not be dispatched.")
                return
            if future is None:
                self._cancel()

    def _validated(self, value):
        if not isinstance(value, dict) or not _source_id(value.get("source_window_id")) or value["source_window_id"] != self.source_window_id:
            raise ValueError("Invalid checker source")
        if value.get("status") not in _STATUSES or value.get("kind") not in _KINDS:
            raise ValueError("Invalid checker status")
        if not isinstance(value.get("reason"), str) or not value["reason"].strip() or len(value["reason"]) > 2000:
            raise ValueError("Invalid checker reason")
        evidence = value.get("evidence")
        if not isinstance(evidence, list) or len(evidence) > 16 or any(not isinstance(item, str) or len(item) > 2000 for item in evidence):
            raise ValueError("Invalid checker evidence")
        unknown_transmission=(value.get("kind")=="multimodal" and value.get("status") in ("unverifiable","cancelled")
            and value.get("call_made") is True and value.get("sent_keyframes") is None and value.get("sent_transcript") is None
            and value.get("attempted_keyframes")==2 and value.get("attempted_transcript") is True
            and value.get("transmission_status")=="attempted_outcome_unknown" and value.get("call_outcome_unknown") is True)
        if type(value.get("call_made")) is not bool or (not unknown_transmission and (type(value.get("sent_keyframes")) is not int or not 0 <= value["sent_keyframes"] <= 2)):
            raise ValueError("Invalid checker transmission metadata")
        if value["sent_keyframes"] and not value["call_made"]:
            raise ValueError("Inconsistent checker transmission metadata")
        if (value["call_made"] or value["sent_keyframes"]) and not (self._visual_authorized and self._had_images):
            raise ValueError("Unauthorized visual response")
        if value["status"] in ("contradiction", "no_contradiction") and value["kind"] == "none":
            raise ValueError("Unsupported checker decision")
        if value["kind"] in ("visual","multimodal") and value["status"] in ("contradiction", "no_contradiction") and value["sent_keyframes"] != 2:
            raise ValueError("Visual decision lacks two image observations")
        if value.get("sent_transcript") and (not value["call_made"] or not self._visual_authorized or not self._had_images):
            raise ValueError("Unauthorized transcript transmission metadata")
        if value["kind"]=="multimodal" and value["status"] in ("contradiction","no_contradiction") and value.get("sent_transcript") is not True:
            raise ValueError("Multimodal decision lacks a transmitted source transcript")
        output = {key: copy.deepcopy(value[key]) for key in ("source_window_id", "status", "kind", "reason", "evidence", "call_made", "sent_keyframes")}
        output.update(never_confirms_truth=True, decision_effect="contradiction_only")
        if unknown_transmission:output["call_outcome_unknown"]=True
        if output["status"] == "cancelled":
            output["evidence"] = []
        # Preserve only bounded public checker fields; never echo arbitrary
        # image, credential, exception or internal-worker fields.
        for key in ("provider", "model", "claimed_color", "observed_color", "garment", "transmission_status",
                    "expression", "claimed_expression", "calculated_left", "calculated_right",
                    "requested_model", "upstream_provider", "decision_choice"):
            if key in value:
                if not isinstance(value[key], str) or len(value[key]) > (200 if key in ("expression", "claimed_expression") else 100):
                    raise ValueError("Invalid checker metadata")
                output[key] = value[key]
        if "input_sha256" in value:
            digest=value["input_sha256"]
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid input fingerprint")
            output["input_sha256"]=digest
        for key in ("sent_transcript","attempted_transcript","model_version_verified"):
            if key in value:
                if type(value[key]) is not bool and not (unknown_transmission and key=="sent_transcript" and value[key] is None):raise ValueError("Invalid decision metadata")
                output[key]=value[key]
        if "decision_probabilities" in value:
            probabilities=value["decision_probabilities"]
            if (not isinstance(probabilities,dict) or set(probabilities)!={"contradiction","no_contradiction","insufficient_evidence"}
                    or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in probabilities.values())
                    or abs(sum(probabilities.values())-1)>1e-3):raise ValueError("Invalid decision probabilities")
            output["decision_probabilities"]=copy.deepcopy(probabilities)
        if "consent_required" in value:
            if type(value["consent_required"]) is not bool:
                raise ValueError("Invalid checker metadata")
            output["consent_required"] = value["consent_required"]
        if "attempted_keyframes" in value:
            if type(value["attempted_keyframes"]) is not int or not 0 <= value["attempted_keyframes"] <= 2:
                raise ValueError("Invalid checker metadata")
            output["attempted_keyframes"] = value["attempted_keyframes"]
        if "elapsed_ms" in value:
            elapsed = value["elapsed_ms"]
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Invalid checker timing")
            output["elapsed_ms"] = elapsed
        return output

    def _settle(self, value):
        with self._lock:
            if self._result is None:
                self._result = value
            return copy.deepcopy(self._result)

    def _discarded(self, status, reason):
        future = self._future
        accounting = None
        if future is not None and future.done() and not future.cancelled():
            try:
                accounting = self._validated(future.result())
            except Exception:
                pass
        value = self._common(status, reason, unknown=accounting is None and self._unknown())
        if accounting is not None:
            for key in ("call_made", "sent_keyframes", "sent_transcript", "attempted_keyframes", "attempted_transcript", "transmission_status"):
                if key in accounting:
                    value[key] = accounting[key]
        return value

    def _cancelled_result(self):
        self._cancel()
        value = self._discarded("cancelled", "The source answer was cancelled; contradiction output was discarded.")
        # Cancellation always supersedes an earlier completed answer result.
        with self._lock:
            self._result = value
        return copy.deepcopy(value)

    def finish(self, transcriptfallback=None):
        """Collect within the original submit deadline, or fail closed promptly."""
        if self._cancelled():
            return self._cancelled_result()
        with self._lock:
            if not self._attempted and self._result is None:
                self.on_transcript(transcriptfallback)
            future = self._future
            result = self._result
        if result is not None:
            if self._cancelled():
                return self._cancelled_result()
            return copy.deepcopy(result)
        if future is None:
            if self._cancelled():
                return self._cancelled_result()
            self._runtime._released(self)
            return self._settle(self._common("unverifiable", "No complete local ASR transcript was available for the source answer."))
        while True:
            if self._cancelled():
                return self._cancelled_result()
            if future.done():
                if self._completed_at is not None and self._completed_at > self._deadline_at:
                    self._stop.owned.set()
                    return self._settle(self._discarded("unverifiable", "The claim-check collection deadline expired; no contradiction was asserted."))
                try:
                    value = self._validated(future.result())
                except Exception:
                    value = self._common("unverifiable", "The claim worker did not return a valid same-answer result; no contradiction was asserted.", unknown=self._unknown())
                if self._cancelled():
                    return self._cancelled_result()
                return self._settle(value)
            remaining = self._deadline_at - time.monotonic()
            if remaining <= 0:
                self._stop.owned.set()
                future.cancel()
                with self._lock:
                    self._images = ()
                return self._settle(self._discarded("unverifiable", "The claim-check collection deadline expired; no contradiction was asserted."))
            self._stop.owned.wait(min(.025, remaining))
