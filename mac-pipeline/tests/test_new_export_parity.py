import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from sunhacks.local_digest import (load_digest_schema,digest_contract,build_digest_row,ensemble_policy,
    combine_ensemble_scores,NEW_FEATURE_VERSION,LEGACY_FEATURE_VERSION,NEW_SCHEMA_SHA256,DigestRow)
from sunhacks.local_stack import LocalDigestStack
from sunhacks.vendor import friend_features_tabular_v3 as v3

ROOT=Path(__file__).resolve().parents[1]


def new_manifest():
    # Exact reviewed serving coefficients; no trained model or dataset is loaded.
    return {'format':'sunhacks-ensemble-v2','feature_version':NEW_FEATURE_VERSION,
        'branches':['tuned_dasheng','xgb_digest'],
        'platt_A':[1.2428574190344905,1.5430329189543752],
        'platt_B':[.18563188430181185,.48678425930137054],
        'weights':[.5379349436053581,.462065056394642],
        'threshold':.4238521751969923,
        'threshold_provenance':'Prior-matched grouped-CV serving threshold; technical test fixture.',
        'digest_model_sha256':'f501ea925d0cc2f8d15776b239d97463963f2a83e4a155303420eaed36f563c1',
        'digest_schema_sha256':NEW_SCHEMA_SHA256,
        'cv_metrics':{'protocol':'technical fixture only'}}


def test_exact_reviewed_schema_and_serving_threshold():
    old=load_digest_schema();new=load_digest_schema(NEW_FEATURE_VERSION)
    assert len(old['feature_names'])==89 and len(new['feature_names'])==88
    assert 'txt_word_count' not in new['feature_names'] and 'txt_question_marks_per100' in new['feature_names']
    assert 'face_upper_motion_p95' in new['feature_names'] and 'face_upper_motion_max' not in new['feature_names']
    assert ensemble_policy(new_manifest())['threshold']==pytest.approx(.4238521751969923)
    with pytest.raises(ValueError):load_digest_schema('v4-unknown')


def test_new_platt_formula_matches_reviewed_serving_oracle():
    m=new_manifest()
    for pd,px in [(0.,1.),(.2,.4),(.5,.5),(.9,.3),(1.,0.)]:
        p=np.clip([pd,px],1e-4,1-1e-4)
        expected=float(np.dot(m['weights'],1/(1+np.exp(-(np.log(p/(1-p))*m['platt_A']+m['platt_B'])))))
        assert combine_ensemble_scores(pd,px,m)==pytest.approx(expected,abs=1e-15)


@pytest.mark.parametrize('bad',[True,0,1,float('nan'),'0.4',None])
def test_invalid_new_threshold_is_never_silently_replaced_by_half(bad):
    m=new_manifest();m['threshold']=bad
    with pytest.raises(ValueError):ensemble_policy(m)


def test_legacy_half_threshold_is_explicit_and_unsupported_pair_rejected():
    old={'format':'sunhacks-ensemble-v1'}
    assert ensemble_policy(old)['threshold']==.5
    old['feature_version']=NEW_FEATURE_VERSION
    with pytest.raises(ValueError):ensemble_policy(old)
    m=new_manifest();m['feature_version']=LEGACY_FEATURE_VERSION
    with pytest.raises(ValueError):ensemble_policy(m)


def test_fixed_lag_head_motion_rates_match_known_linear_motion_at_seven_and_long_answers():
    for duration in [7.,27.,81.]:
        ts=(np.arange(128)+.5)*duration/128
        boxes=np.column_stack([400+100*ts,np.full(128,300),np.full(128,200)])
        f=v3.face_features(np.zeros((128,16,16,3),np.uint8),{'boxes_cxcys':boxes,'timestamps':ts,'face_detect_rate':1.})
        assert f['face_head_speed_mean']==pytest.approx(.5,abs=1e-6)
        assert f['face_head_speed_p95']==pytest.approx(.5,abs=1e-6)


def test_exact_v3_row_has_eighty_eight_columns_and_new_feature_names():
    ts=np.linspace(0,7,128);frames=np.zeros((128,384,384,3),np.uint8)
    meta={'boxes_cxcys':np.tile([192.,192.,200.],(128,1)), 'timestamps':ts,'face_detect_rate':1.}
    q=np.ones(1024);intents=np.eye(33,1024)
    row=build_digest_row(transcript='Did I? No, I did not do it?',audio=np.zeros(112000,np.float32),
        duration_s=7.,face_crops=frames,face_metadata=meta,emotion_probabilities=np.full(9,1/9),
        qwen_embedding=q,intent_embeddings=intents,feature_version=NEW_FEATURE_VERSION)
    assert row.values.shape==(1,88) and row.feature_version==NEW_FEATURE_VERSION
    assert tuple(load_digest_schema(NEW_FEATURE_VERSION)['feature_names'])==row.feature_names
    values=row.public_features()
    assert 'txt_word_count' not in values and values['txt_question_marks_per100']>0


def setup_loader(tmp_path,monkeypatch,*,bad_model_binding=False):
    bp=tmp_path/'model.json';bp.write_bytes(b'{"technical_fixture":true}')
    modelsha=hashlib.sha256(bp.read_bytes()).hexdigest();m=new_manifest()
    m['digest_model_sha256']='0'*64 if bad_model_binding else modelsha
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(m))
    calls=[]
    class Booster:
        feature_names=None
        def __init__(self,params):calls.append('construct')
        def load_model(self,data):pass
        def set_param(self,params):pass
        def num_features(self):return 88
        def save_config(self):return json.dumps({'learner':{'objective':{'name':'binary:logistic'}}})
        def predict(self,matrix):return np.array([.4])
    monkeypatch.setitem(sys.modules,'xgboost',SimpleNamespace(Booster=Booster,DMatrix=lambda values,feature_names:values))
    c=LocalDigestStack(mp,bp,feature_version=NEW_FEATURE_VERSION,
        expected_manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),expected_model_sha256=modelsha)
    return c,m,calls


def test_v2_stack_reports_exported_policy_and_feature_version(tmp_path,monkeypatch):
    c,m,calls=setup_loader(tmp_path,monkeypatch);c.prepare()
    row=DigestRow(tuple(c.schema['feature_names']),np.ones((1,88),np.float32),(),NEW_FEATURE_VERSION)
    r=c.infer(row,.6,source_window_id=5)
    assert r['tabular']['model_id']=='teammate/xgb_digest_88'
    assert r['stack']['threshold']==m['threshold'] and r['stack']['threshold_provenance']==m['threshold_provenance']
    assert r['stack']['feature_version']==NEW_FEATURE_VERSION and r['stack']['cv_metrics']==m['cv_metrics']


def test_v2_hash_binding_mismatch_rejected_before_model_backend(tmp_path,monkeypatch):
    c,_,calls=setup_loader(tmp_path,monkeypatch,bad_model_binding=True)
    with pytest.raises(ValueError,match='binding'):c.prepare()
    assert not calls
