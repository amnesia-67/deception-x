"""Offline deployment must never execute withdrawn checkpoints or live capture."""
from concurrent.futures import Future
import threading

import pytest

from sunhacks.pipeline import Pipeline
from sunhacks import heads, teammate_fusion

QUARANTINE = {"deployment":{"artifacts_status":"awaiting_retrained_weights",
                            "artifact_reason":"Withdrawn artifacts await replacement."}}


def test_quarantine_blocks_teammate_head_before_weights_or_features(monkeypatch):
    monkeypatch.setattr(teammate_fusion,"_load",lambda cfg:pytest.fail("Withdrawn checkpoint loaded"))
    monkeypatch.setattr(teammate_fusion,"_array",lambda *args:pytest.fail("Quarantined features converted"))
    result = teammate_fusion.run_teammate_fusion({},QUARANTINE)
    assert result["status"] == "unavailable"
    assert result["raw_score"] is None
    assert result["reason"] == "Withdrawn artifacts await replacement."


@pytest.mark.parametrize("function,inputs",[(heads.run_fusion,{}),(heads.run_tabular,{}),(heads.run_stack,{})])
def test_quarantine_blocks_all_reference_heads_before_artifact_access(monkeypatch,function,inputs):
    monkeypatch.setattr(heads.Path,"read_text",lambda *args:pytest.fail("Quarantined artifact read"))
    result = function(inputs,QUARANTINE)
    assert result["status"] == "unavailable"
    assert "Withdrawn artifacts" in result["reason"]


def test_remote_offline_blocks_live_before_capture_or_runtime(monkeypatch):
    from sunhacks.capture import LiveCapture
    monkeypatch.setattr(LiveCapture,"start",lambda self:pytest.fail("Offline session opened sensors"))
    pipeline = Pipeline({"runtime_mode":"remote_fast",**QUARANTINE,
                         "remote_vision":{"status":"offline_by_user"}})
    with pytest.raises(RuntimeError,match="unavailable"):
        pipeline.start("live")
    assert pipeline.mode == "stopped"
    assert pipeline.capture is None
    assert pipeline.fast_runtime is None


def test_preparation_only_loads_local_runtime_and_exposes_partial_state(monkeypatch):
    from sunhacks import fast_runtime
    calls=[]
    class LocalOnlyRuntime:
        def __init__(self,config,vision_runner=None,on_progress=None):
            assert vision_runner is None
            calls.append("construct")
        def snapshot(self):return {"active_jobs":[],"active_prepares":0}
        def prepare(self):
            calls.append("prepare_local")
            return {"status":"partial","components":{},"local_weights_loaded":True,
                    "kernels_warmed":False,"remote_verified":False}
        def close(self):return {"status":"closed"}
    monkeypatch.setattr(fast_runtime,"FastRuntime",LocalOnlyRuntime)
    pipeline=Pipeline({"runtime_mode":"remote_fast",**QUARANTINE,
                       "remote_vision":{"status":"offline_by_user","host":"192.168.0.224","port":5002}})
    pipeline.prepare_runtime()
    pipeline.prepare_thread.join(1)
    state=pipeline.state()
    assert state["mode"] == "stopped" and pipeline.capture is None
    assert state["runtime"]["local_weights_loaded"] is True
    assert state["runtime"]["live_ready"] is False
    assert state["runtime"]["remote"]["encoder_verified"] is False
    assert calls == ["construct","prepare_local"]
    pipeline.close()


@pytest.mark.parametrize("change",["session","window","stop","closed","cancelled","valid"])
def test_late_emotion_requires_matching_session_window_and_stop_state(change):
    pipeline=Pipeline()
    pipeline.mode="live"
    pipeline.session_id="old"
    pipeline.latest={"window":{"id":1},"branches":{"emotion":{"status":"unavailable"}}}
    source_stop=threading.Event()
    future=Future()
    pipeline._watch_emotion(future,"old",1,source_stop)
    if change=="session":pipeline.session_id="new"
    if change=="window":pipeline.latest["window"]["id"]=2
    if change=="stop":source_stop.set()
    if change=="closed":pipeline.closed=True
    result={"source_window_id":1,"status":"cancelled" if change=="cancelled" else "base"}
    future.set_result(result)
    assert pipeline.latest["branches"]["emotion"]["status"] == ("base" if change=="valid" else "unavailable")


def test_baseline_profile_cannot_prepare_duplicate_resident_runtime():
    pipeline=Pipeline()
    with pytest.raises(RuntimeError,match="remote_fast"):
        pipeline.prepare_runtime()
    assert pipeline.fast_runtime is None


@pytest.mark.parametrize("status",["awaiting_deployment_export","awaiting_remote_verification"])
def test_new_pending_deployment_status_never_reenables_withdrawn_heads(monkeypatch,status):
    config={"deployment":{"artifacts_status":status,"artifact_reason":"Matching exports pending."}}
    monkeypatch.setattr(teammate_fusion,"_load",lambda cfg:pytest.fail("Old head loaded after status update"))
    assert teammate_fusion.run_teammate_fusion({},config)["raw_score"] is None
    assert "Matching exports pending" in heads.run_fusion({},config)["reason"]


def test_reachable_unverified_service_is_not_mislabeled_offline_or_accepted_as_encoder():
    pipeline=Pipeline({"runtime_mode":"remote_fast",
        "remote_vision":{"status":"protocol_unverified","host":"192.168.0.224","port":5002,
                         "reachable":True,"health_elapsed_ms":18.943,
                         "advertised_video_model":"vjepa2_vitl_lora_5fold"},
        "deployment":{"artifacts_status":"awaiting_deployment_export",
                      "runtime_blocker":"Reachable service identifies V-JEPA2; approved2.1 export and serving source are pending."}})
    value=pipeline.state()["runtime"]
    assert value["remote"]["reachable"] is True
    assert value["remote"]["advertised_video_model"]=="vjepa2_vitl_lora_5fold"
    assert value["live_ready"] is False and value["remote"]["encoder_verified"] is False
    assert "service is off" not in value["reason"]
    with pytest.raises(RuntimeError,match="approved2.1"):
        pipeline.start("live")
    assert pipeline.capture is None
