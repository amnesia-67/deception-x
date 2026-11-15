import hashlib
import json
import sys
import types

import numpy as np
import pytest

from sunhacks.heads import BRANCH_ORDER, run_fusion, run_tabular, run_stack, fusion_module
from sunhacks.models import CONTRACT, DIMS, MODEL_IDS, POOLING


def artifact(tmp_path, name, architecture, filename, content=b"fixture", **metadata):
    path = tmp_path / name
    path.mkdir()
    (path / filename).write_bytes(content)
    manifest = {"input_contract":CONTRACT,"architecture":architecture,
                "sha256":{filename:hashlib.sha256(content).hexdigest()},
                "class_names":["class_a","class_b"],**metadata}
    (path / "handoff.json").write_text(json.dumps(manifest))
    return {"models":{name:{"path":str(path)}}}, manifest


def fusion_fixture(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    model = fusion_module()
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
        model[-1].bias.copy_(torch.tensor([0.,1.]))
    path = tmp_path / "fusion"
    path.mkdir()
    safetensors.save_file(model.state_dict(),str(path / "model.safetensors"))
    inputs = [{"branch":name,"model_id":MODEL_IDS[name],"base_revision":"reviewed-base-sha",
               "output_dim":DIMS[name],"pooling":POOLING[name],"adapter_sha256":None} for name in BRANCH_ORDER]
    config = {"models":{"fusion":{"path":str(path)}},"runtime":{"text_scope":"video_window"}}
    from sunhacks.models import input_settings
    manifest = {"input_contract":CONTRACT,"architecture":"sunhacks-fusion-mlp-v1","inputs":inputs,
                "preprocessing_settings":input_settings(config),
                "text_scope":"video_window","class_names":["class_a","class_b"],
                "sha256":{"model.safetensors":hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()}}
    (path / "handoff.json").write_text(json.dumps(manifest))
    branches = {name:{"status":"base","model_id":MODEL_IDS[name],"revision":"reviewed-base-sha",
                      "pooling":POOLING[name],"_embedding":np.zeros(DIMS[name],np.float32)} for name in BRANCH_ORDER}
    return branches,config,path,manifest


def test_missing_heads_are_unavailable():
    assert run_fusion({}, {})["status"] == "unavailable"
    assert run_tabular({}, {})["status"] == "unavailable"
    assert run_stack({}, {})["status"] == "unavailable"


def test_reference_fusion_valid_shape_and_uncalibrated(tmp_path):
    branches,config,_,_ = fusion_fixture(tmp_path)
    result = run_fusion(branches,config)
    assert result["status"] == "trained", result
    assert result["calibrated"] is False
    assert result["class_names"] == ["class_a","class_b"]
    assert result["logits"] == [0.,1.]
    assert result["probabilities"][1] == pytest.approx(.7310586)


@pytest.mark.parametrize("failure",["mock","dimension","revision","scope","hash","sample_fps","preprocessing_missing"])
def test_fusion_rejects_incompatible_inputs(tmp_path,failure):
    branches,config,path,_ = fusion_fixture(tmp_path)
    if failure == "mock": branches["vision"]["status"] = "mock"
    elif failure == "dimension": branches["emotion"]["_embedding"] = np.zeros(768)
    elif failure == "revision": branches["vision"]["revision"] = "different-sha"
    elif failure == "scope": config["runtime"]["text_scope"] = "session_context"
    elif failure == "hash": (path/"model.safetensors").write_bytes(b"modified")
    elif failure == "sample_fps": config["sample_fps"] = 16
    elif failure == "preprocessing_missing":
        manifest = json.loads((path/"handoff.json").read_text())
        manifest.pop("preprocessing_settings")
        (path/"handoff.json").write_text(json.dumps(manifest))
    assert run_fusion(branches,config)["status"] == "unavailable"


def test_tabular_requires_exact_names_and_real_values(tmp_path,monkeypatch):
    name = "audio.rms_dbfs"
    config,_ = artifact(tmp_path,"tabular","sunhacks-xgboost-diagnostics-v1","model.json",
                        feature_names=[name],missingness_policy="reject")
    class FakeBooster:
        feature_names = [name]
        def load_model(self,path): pass
        def save_config(self): return json.dumps({"learner":{"objective":{"name":"binary:logistic"}}})
        def predict(self,data): return np.array([.25])
    monkeypatch.setitem(sys.modules,"xgboost",types.SimpleNamespace(Booster=FakeBooster,DMatrix=lambda values,feature_names:values))
    result = run_tabular({"audio":{"status":"ok","rms_dbfs":-20.}},config)
    assert result["status"] == "trained", result
    assert result["probabilities"] == [.75,.25]
    assert result["calibrated"] is False
    assert run_tabular({"audio":{"status":"missing","rms_dbfs":None}},config)["status"] == "unavailable"
    FakeBooster.feature_names = ["audio.peak_dbfs"]
    assert run_tabular({"audio":{"status":"ok","rms_dbfs":-20.}},config)["status"] == "unavailable"


def test_tabular_rejects_unimplemented_feature(tmp_path):
    config,_ = artifact(tmp_path,"tabular","sunhacks-xgboost-diagnostics-v1","model.json",
                        feature_names=["face.au20"],missingness_policy="reject")
    result = run_tabular({},config)
    assert result["status"] == "unavailable"
    assert "implemented" in result["reason"]


def test_stack_requires_all_branches_and_class_order(tmp_path):
    params = {"inputs":["fusion","tabular"],"input_semantics":"class_1_probability",
              "coefficients":[1.,1.],"intercept":-1.,"platt_a":1.,"platt_b":0.}
    config,_ = artifact(tmp_path,"stack","sunhacks-stack-logistic-v1","parameters.json",
                        content=json.dumps(params).encode(),missingness_policy="reject")
    branches = {name:{"status":"trained","class_names":["class_a","class_b"],"probabilities":[.5,.5]} for name in ("fusion","tabular")}
    result = run_stack(branches,config)
    assert result["status"] == "trained", result
    assert result["probabilities"] == [.5,.5]
    assert result["calibrated"] is False
    del branches["tabular"]
    assert run_stack(branches,config)["status"] == "unavailable"


def test_stack_rejects_nonfinite_or_invalid_probability(tmp_path):
    params = {"inputs":["fusion"],"input_semantics":"class_1_probability",
              "coefficients":[1.],"intercept":0.,"platt_a":1.,"platt_b":0.}
    config,_ = artifact(tmp_path,"stack","sunhacks-stack-logistic-v1","parameters.json",
                        content=json.dumps(params).encode(),missingness_policy="reject")
    result = run_stack({"fusion":{"status":"trained","class_names":["class_a","class_b"],"probabilities":[float("nan"),.5]}},config)
    assert result["status"] == "unavailable"
