"""Real owned worker lifecycle with fake checkers; no APIs, models or sensors."""
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from sunhacks.claim_runtime import ClaimRuntime


def transcript(window=7, **changes):
    return {"source_window_id": window, "source": "local_asr", "scope": "video_window",
            "status": "base", "text": "Two plus two equals four.", **changes}


def result(window=7, **changes):
    return {"source_window_id": window, "status": "no_contradiction", "kind": "math",
            "reason": "No arithmetic contradiction; this does not establish truth.", "evidence": ["2 + 2 = 4"],
            "call_made": False, "sent_keyframes": 0, **changes}


class Checker:
    deadline = .04

    def __init__(self, check=None):
        self.calls = []
        self.check_fn = check

    def check(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return self.check_fn(text, **kwargs) if self.check_fn else result(kwargs["source_window_id"])


def test_default_checker_import_and_construction_are_lazy(monkeypatch):
    created = []
    checker = Checker()
    def factory(config):
        created.append(config)
        return checker
    monkeypatch.setitem(sys.modules, "sunhacks.claim_check", SimpleNamespace(ClaimChecker=factory))
    runtime = ClaimRuntime({"deadline_seconds": .1})
    task = runtime.begin(source_window_id=7)
    assert created == [] and checker.calls == []
    task.on_transcript(transcript())
    assert task.finish()["status"] == "no_contradiction"
    assert created == [{"deadline_seconds": .1}]
    runtime.close()


def test_callbacks_return_immediately_and_two_checks_run_in_parallel():
    release = threading.Event()
    both_entered = threading.Event()
    counter_lock = threading.Lock()
    entered = []
    def check(text, **kwargs):
        with counter_lock:
            entered.append(kwargs["source_window_id"])
            if len(entered) == 2:
                both_entered.set()
        release.wait(1)
        return result(kwargs["source_window_id"])
    checker = Checker(check)
    runtime = ClaimRuntime({}, checker)
    first = runtime.begin(source_window_id=7)
    second = runtime.begin(source_window_id=8)
    try:
        started = time.monotonic()
        first.on_transcript(transcript())
        second.on_transcript(transcript(8))
        assert time.monotonic() - started < .1
        assert both_entered.wait(.5)
        release.set()
        assert first.finish()["status"] == second.finish()["status"] == "no_contradiction"
    finally:
        release.set()
        runtime.close()


def test_duplicate_callback_and_finish_fallback_never_resubmit_or_change_words():
    checker = Checker()
    runtime = ClaimRuntime({}, checker)
    task = runtime.begin(source_window_id=7)
    task.on_transcript(transcript())
    task.on_transcript(transcript(text="Changed text"))
    first = task.finish(transcript(text="Fallback text"))
    second = task.finish(transcript(text="Another text"))
    assert first == second
    first["evidence"].append("caller mutation")
    assert "caller mutation" not in task.finish()["evidence"]
    assert len(checker.calls) == 1 and checker.calls[0][0] == transcript()["text"]
    runtime.close()


@pytest.mark.parametrize("changes", [
    {"status": "partial"}, {"status": "missing"}, {"status": "error"}, {"status": []},
    {"text": ""}, {"text": "   "}, {"text": "x" * 4001}, {"text": None},
    {"source_window_id": 8}, {"source_window_id": True}, {"source_window_id": 7.0},
    {"source_window_id": "7"}, {"source": "manual"}, {"scope": "session"}, {"scope": []},
    {"truncated": True},
])
def test_invalid_or_partial_transcript_cannot_dispatch_or_be_replaced_by_fallback(changes):
    checker = Checker()
    runtime = ClaimRuntime({}, checker)
    task = runtime.begin(source_window_id=7)
    task.on_transcript(transcript(**changes))
    value = task.finish(transcript())
    assert value["status"] == "unverifiable" and value["source_window_id"] == 7
    assert value["call_made"] is False and checker.calls == []
    runtime.close()


@pytest.mark.parametrize("status", ["base", "available"])
def test_finish_can_dispatch_one_valid_same_asr_fallback(status):
    checker = Checker()
    runtime = ClaimRuntime({}, checker)
    task = runtime.begin(source_window_id=7)
    assert task.finish(transcript(status=status))["status"] == "no_contradiction"
    assert task.finish(transcript())["status"] == "no_contradiction"
    assert len(checker.calls) == 1
    runtime.close()


def test_missing_fallback_finishes_promptly_without_dispatch():
    checker = Checker()
    runtime = ClaimRuntime({}, checker)
    task = runtime.begin(source_window_id=7)
    started = time.monotonic()
    assert task.finish()["status"] == "unverifiable"
    assert time.monotonic() - started < .05 and checker.calls == []
    runtime.close()


def test_unauthorized_images_are_not_inspected_retained_or_passed_but_math_runs():
    class UnreadableImages:
        def __iter__(self):
            pytest.fail("Unauthorized images were inspected")
    checker = Checker()
    runtime = ClaimRuntime({}, checker)
    task = runtime.begin(images=UnreadableImages(), visual_authorized=False, source_window_id=7)
    assert task._images == ()
    assert task.finish(transcript())["kind"] == "math"
    assert checker.calls[0][1]["images"] == ()
    assert checker.calls[0][1]["visual_authorized"] is False
    runtime.close()


@pytest.mark.parametrize("authorization,images", [(1, ("a", "b")), (True, ["a"]), (True, ["a", "b", "c"]), (True, ["a", object()])])
def test_invalid_authorization_or_image_bounds_fail_without_worker(authorization, images):
    checker = Checker()
    runtime = ClaimRuntime({}, checker)
    task = runtime.begin(images=images, visual_authorized=authorization, source_window_id=7)
    assert task.finish(transcript())["status"] == "unverifiable"
    assert task._images == () and checker.calls == []
    runtime.close()


def test_two_admitted_tasks_bound_pending_images_without_any_executor_queue():
    checker = Checker()
    runtime = ClaimRuntime({}, checker)
    first = runtime.begin(images=("a", "b"), visual_authorized=True, source_window_id=7)
    second = runtime.begin(images=("c", "d"), visual_authorized=True, source_window_id=8)
    third = runtime.begin(images=("e", "f"), visual_authorized=True, source_window_id=9)
    assert third.finish(transcript(9))["status"] == "unverifiable"
    assert "occupied" in third.finish()["reason"] and third._images == () and checker.calls == []
    first.finish()  # Missing transcript releases an unused slot.
    fourth = runtime.begin(source_window_id=10)
    assert fourth.finish(transcript(10))["status"] == "no_contradiction"
    runtime.close()


def test_source_stop_before_dispatch_prevents_check():
    stopped = threading.Event()
    stopped.set()
    checker = Checker()
    runtime = ClaimRuntime({}, checker)
    task = runtime.begin(source_window_id=7, stop_event=stopped)
    task.on_transcript(transcript())
    assert task.finish()["status"] == "cancelled" and checker.calls == []
    runtime.close()


def test_stop_running_worker_is_prompt_unknown_and_does_not_release_busy_slot():
    release = threading.Event()
    entered = threading.Event()
    def check(text, **kwargs):
        entered.set()
        release.wait(1)
        return result(kwargs["source_window_id"], kind="visual", call_made=True, sent_keyframes=2)
    checker = Checker(check)
    runtime = ClaimRuntime({}, checker)
    stop = threading.Event()
    first = runtime.begin(images=("a", "b"), visual_authorized=True, source_window_id=7, stop_event=stop)
    reserved = runtime.begin(source_window_id=8)
    try:
        first.on_transcript(transcript())
        assert entered.wait(.5)
        stop.set()
        started = time.monotonic()
        value = first.finish()
        assert time.monotonic() - started < .1
        assert value["status"] == "cancelled" and value["evidence"] == []
        assert value["call_made"] is None and value["sent_keyframes"] is None
        assert value["call_outcome_unknown"] is True and value["attempted_keyframes"] == 2
        third = runtime.begin(source_window_id=9)
        assert "occupied" in third.finish()["reason"]
        release.set()
        assert first._future.result(timeout=.5)["status"] == "no_contradiction"
        assert first.finish()["status"] == "cancelled"
    finally:
        release.set()
        runtime.close()


def test_finish_deadline_is_anchored_to_submission_and_late_result_cannot_replace_it():
    release = threading.Event()
    entered = threading.Event()
    def check(text, **kwargs):
        entered.set()
        release.wait(1)
        return result(kwargs["source_window_id"])
    checker = Checker(check)
    checker.deadline = .01
    runtime = ClaimRuntime({}, checker)
    task = runtime.begin(images=("a", "b"), visual_authorized=True, source_window_id=7)
    try:
        task.on_transcript(transcript())
        assert entered.wait(.5)
        threading.Event().wait(.24)
        started = time.monotonic()
        value = task.finish()
        assert time.monotonic() - started < .08
        assert value["status"] == "unverifiable" and "deadline" in value["reason"]
        assert value["call_made"] is None and value["sent_keyframes"] is None
        assert task._stop.is_set() and task._images == ()
        release.set()
        task._future.result(timeout=.5)
        assert task.finish() == value
    finally:
        release.set()
        runtime.close()


def test_finish_waits_only_remaining_deadline_and_keeps_actual_worker_slot():
    release = threading.Event()
    checker = Checker(lambda text, **kwargs: (release.wait(1), result(kwargs["source_window_id"]))[1])
    checker.deadline = .01
    runtime = ClaimRuntime({}, checker)
    first = runtime.begin(source_window_id=7)
    second = runtime.begin(source_window_id=8)
    try:
        first.on_transcript(transcript())
        started = time.monotonic()
        value = first.finish()
        assert .15 < time.monotonic() - started < .5
        assert value["status"] == "unverifiable"
        assert "occupied" in runtime.begin(source_window_id=9).finish()["reason"]
    finally:
        release.set()
        runtime.close()


def test_completed_but_late_response_is_discarded_when_finish_was_not_waiting():
    def check(text, **kwargs):
        threading.Event().wait(.24)
        return result(kwargs["source_window_id"])
    checker = Checker(check)
    checker.deadline = .01
    runtime = ClaimRuntime({}, checker)
    task = runtime.begin(source_window_id=7)
    task.on_transcript(transcript())
    task._future.result(timeout=.5)
    value = task.finish()
    assert value["status"] == "unverifiable" and "deadline" in value["reason"]
    assert value["evidence"] == []
    runtime.close()


def test_slow_lazy_worker_start_does_not_dispatch_checker_after_deadline(monkeypatch):
    checker = Checker()
    runtime = ClaimRuntime({}, checker)
    runtime.deadline = .01
    def slow_start():
        threading.Event().wait(.24)
        return checker
    monkeypatch.setattr(runtime, "_get_checker", slow_start)
    task = runtime.begin(source_window_id=7)
    task.on_transcript(transcript())
    task._future.result(timeout=.5)
    value = task.finish()
    assert value["status"] == "unverifiable" and checker.calls == []
    assert value["call_made"] is False
    runtime.close()


@pytest.mark.parametrize("changes", [
    {"source_window_id": 8}, {"source_window_id": True}, {"source_window_id": 7.0},
    {"status": "true"}, {"status": "truthful"}, {"status": []}, {"kind": "honesty"},
    {"call_made": 1}, {"sent_keyframes": True}, {"evidence": "not a list"}, {"reason": ""},
    {"kind": "visual", "call_made": True, "sent_keyframes": 2},
])
def test_wrong_source_unsupported_or_unauthorized_checker_result_fails_closed(changes):
    checker = Checker(lambda text, **kwargs: result(**changes))
    runtime = ClaimRuntime({}, checker)
    value = runtime.begin(source_window_id=7).finish(transcript())
    assert value["status"] == "unverifiable" and value["source_window_id"] == 7
    assert value["evidence"] == [] and value["never_confirms_truth"] is True
    runtime.close()


def test_checker_exceptions_are_generic_and_do_not_expose_details():
    def fail(text, **kwargs):
        raise RuntimeError("private fixture secret")
    runtime = ClaimRuntime({}, Checker(fail))
    value = runtime.begin(source_window_id=7).finish(transcript())
    assert value["status"] == "unverifiable" and "private fixture" not in str(value)
    runtime.close()


def test_only_allowlisted_bounded_checker_output_is_published():
    checker = Checker(lambda text, **kwargs: result(private_image="secret pixels", api_key="secret key", true=True,
                                                   never_confirms_truth=False, decision_effect="truth"))
    runtime = ClaimRuntime({}, checker)
    value = runtime.begin(source_window_id=7).finish(transcript())
    assert value["status"] == "no_contradiction"
    assert not {"private_image", "api_key", "true"}.intersection(value)
    assert value["never_confirms_truth"] is True and value["decision_effect"] == "contradiction_only"
    runtime.close()


def test_cancel_after_completed_result_discards_evidence_but_preserves_call_accounting():
    checker = Checker(lambda text, **kwargs: result(kind="visual", call_made=True, sent_keyframes=2,
                                                   attempted_keyframes=2, transmission_status="response_received"))
    runtime = ClaimRuntime({}, checker)
    stop = threading.Event()
    task = runtime.begin(images=("a", "b"), visual_authorized=True, source_window_id=7, stop_event=stop)
    assert task.finish(transcript())["status"] == "no_contradiction"
    stop.set()
    cancelled = task.finish()
    assert cancelled["status"] == "cancelled" and cancelled["evidence"] == []
    assert cancelled["call_made"] is True and cancelled["sent_keyframes"] == 2
    runtime.close()


def test_close_signals_owned_tasks_without_waiting_and_rejects_future_answers():
    entered = threading.Event()
    exited = threading.Event()
    def check(text, **kwargs):
        entered.set()
        while not kwargs["stop_event"].is_set():
            threading.Event().wait(.005)
        exited.set()
        return result(kwargs["source_window_id"])
    checker = Checker(check)
    runtime = ClaimRuntime({}, checker)
    first = runtime.begin(source_window_id=7)
    second = runtime.begin(source_window_id=8)
    first.on_transcript(transcript())
    assert entered.wait(.5)
    started = time.monotonic()
    runtime.close()
    runtime.close()
    assert time.monotonic() - started < .1
    assert first.finish()["status"] == second.finish()["status"] == "cancelled"
    assert exited.wait(.5)
    assert runtime.begin(source_window_id=9).finish(transcript(9))["status"] == "cancelled"
    assert len(checker.calls) == 1


@pytest.mark.parametrize("source", [True, 7.0, "7", -1, 2**53])
def test_invalid_begin_source_is_rejected(source):
    runtime = ClaimRuntime({}, Checker())
    with pytest.raises(ValueError):
        runtime.begin(source_window_id=source)
    runtime.close()
