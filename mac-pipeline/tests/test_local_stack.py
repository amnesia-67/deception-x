"""Hash/schema and missing-value tests with a fake local XGBoost backend."""
import hashlib
import json
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from sunhacks.local_stack import LocalDigestStack
from sunhacks.local_digest import DigestRow,load_digest_schema,combine_ensemble_scores


def setup(tmp_path,monkeypatch):
    manifest={"format":"sunhacks-ensemble-v1","branches":["tuned_dasheng","xgb_digest"],
              "platt_A":[1.2,1.4],"platt_B":[.2,.4],"weights":[.6,.4]}
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    bp=tmp_path/'model.json';bp.write_bytes(b'{"technical_model_placeholder":true}')
    calls=[]
    class Booster:
        feature_names=None
        def __init__(self,params):calls.append(("construct",params))
        def load_model(self,data):calls.append(("load",data));assert isinstance(data,bytearray)
        def set_param(self,params):pass
        def num_features(self):return 89
        def save_config(self):return json.dumps({"learner":{"objective":{"name":"binary:logistic"}}})
        def predict(self,matrix):calls.append(("predict",matrix));return np.array([.4])
    def dmatrix(values,feature_names):return {"values":values.copy(),"names":feature_names}
    monkeypatch.setitem(sys.modules,'xgboost',SimpleNamespace(Booster=Booster,DMatrix=dmatrix))
    loader=LocalDigestStack(mp,bp,expected_manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
                          expected_model_sha256=hashlib.sha256(bp.read_bytes()).hexdigest())
    return loader,manifest,mp,bp,calls


def test_hashes_checked_before_backend_receives_bytes(tmp_path,monkeypatch):
    loader,manifest,mp,bp,calls=setup(tmp_path,monkeypatch)
    bp.write_bytes(b'altered')
    with pytest.raises(ValueError,match="hash"):
        loader.prepare()
    assert calls==[]


def test_exact_order_and_nan_missingness_survive_to_local_booster(tmp_path,monkeypatch):
    loader,manifest,mp,bp,calls=setup(tmp_path,monkeypatch)
    assert loader.prepare()["locality"]=="mac"
    names=tuple(load_digest_schema()['feature_names'])
    values=np.ones((1,89),np.float32);values[0,10]=np.nan
    row=DigestRow(names,values,(names[10],))
    result=loader.infer(row,.7,source_window_id=5)
    sent=[value for kind,value in calls if kind=='predict'][0]
    assert np.isnan(sent['values'][0,10]) and sent['names']==list(names)
    assert result['stack']['raw_score']==pytest.approx(combine_ensemble_scores(.7,.4,manifest))
    assert result['stack']['not_used_modalities']==['vision']
    assert result['stack']['calibrated'] is False
    assert result['tabular']['source_window_id']==5
    reversed_row=DigestRow(tuple(reversed(names)),values,())
    with pytest.raises(ValueError,match="schema"):
        loader.infer(reversed_row,.7)
    loader.close()
    assert loader.model is None
