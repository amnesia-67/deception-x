import pytest
from sunhacks import remote_jepa_classifier as remote


def response():
    return {'status':'success','model':remote.MODEL_LABEL,'model_id':remote.MODEL_ID,
        'manifest_sha256':remote.MANIFEST_SHA256,'contract':dict(remote.CONTRACT),
        'classifier_only':True,'calibrated':False,'device':'cuda','raw_score':.5,
        'per_fold_probability':[.5]*5,'temporal_attention_32':[.03125]*32,
        'prep_ms':1.,'inference_ms':100.,'n_frames_received':128}


def test_actual_remote_crop_failure_is_visible_to_callers():
    client=remote.RemoteJepa21Classifier('192.0.2.1');value=response()
    value['face']={'face_detect_rate':0.,'centre_crop_fallback':True,'private':'not forwarded'}
    result=client._validate(value,128)
    assert result['face']=={'face_detect_rate':0.,'centre_crop_fallback':True}


@pytest.mark.parametrize('bad',[True,[],{}, {'face_detect_rate':float('nan'),'centre_crop_fallback':False},
    {'face_detect_rate':1.1,'centre_crop_fallback':False},
    {'face_detect_rate':True,'centre_crop_fallback':False},
    {'face_detect_rate':1.,'centre_crop_fallback':1}])
def test_invalid_crop_metadata_is_rejected(bad):
    client=remote.RemoteJepa21Classifier('192.0.2.1');value=response();value['face']=bad
    with pytest.raises(remote.RemoteClassifierError,match='face-crop'):client._validate(value,128)
