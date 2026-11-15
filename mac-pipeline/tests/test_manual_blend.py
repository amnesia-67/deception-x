"""Manual weighting checks: no sensors, network or participant input."""
import copy
import pytest
from sunhacks.manual_blend import blend, checked_weights
from sunhacks.showcase import ShowcaseSession
from test_showcase import real_latest
from test_split_runtime import setup, inputs

WEIGHTS = {'dasheng': .5, 'vision': .25, 'emotion': .25}


def branches():
    return {
        'dasheng': {'status': 'trained', 'source_window_id': 1, 'raw_score': .85, '_pilot_decision': {
            'status': 'trained_classifier', 'profile': 'webcam_pilot', 'source_window_id': 1,
            'raw_score': .8, 'threshold': .48}},
        'vision': {'status': 'trained_classifier', 'source_window_id': 1,
                   'raw_score': .9, 'calibrated_probability': .2, 'threshold': .52},
        'emotion': {'status': 'base', 'source_window_id': 1, 'trained_classifier': {
            'status': 'trained', 'profile': 'webcam_pilot', 'raw_score': .9,
            'calibrated_probability': .4, 'threshold': .50}}}


def test_manual_blend_uses_calibrated_scores_and_matching_weighted_cutoffs():
    source = branches(); original = copy.deepcopy(source)
    result = blend(source, WEIGHTS, 1)
    assert source == original
    assert result['raw_score'] == pytest.approx(.55)
    assert result['threshold'] == pytest.approx(.495)
    assert result['status'] == 'experimental_blend' and result['calibrated'] is False
    assert result['weights'] == WEIGHTS and result['manual_blend']
    assert blend(source, {'dasheng': 1., 'vision': 0., 'emotion': 0.}, 1)['raw_score'] == .8


@pytest.mark.parametrize('weights', [None, {}, {'dasheng': .5},
    {**WEIGHTS, 'vision': float('nan')}, {**WEIGHTS, 'emotion': -.2},
    {**WEIGHTS, 'dasheng': True}, {**WEIGHTS, 'dasheng': .7}])
def test_invalid_weights_fail_explicitly(weights):
    with pytest.raises(ValueError): checked_weights({'weights': weights})


@pytest.mark.parametrize('name', ['dasheng', 'vision', 'emotion'])
@pytest.mark.parametrize('fault', ['missing', 'stale', 'calibration_missing', 'nonfinite'])
def test_no_missing_stale_or_invalid_model_can_silently_change_weights(name, fault):
    source = branches()
    if fault == 'missing': source.pop(name)
    elif fault == 'stale': source[name]['source_window_id'] = 2
    else:
        item = source[name]['_pilot_decision'] if name == 'dasheng' else source[name]['trained_classifier'] if name == 'emotion' else source[name]
        key = 'raw_score' if name == 'dasheng' else 'calibrated_probability'
        if fault == 'calibration_missing': item.pop(key)
        else: item[key] = float('nan')
    result = blend(source, WEIGHTS, 1)
    assert result['status'] == 'unavailable' and result['raw_score'] is None
    assert result['weights'] == WEIGHTS


def test_showcase_uses_blend_and_preserves_contradiction_only_override():
    latest = real_latest(); latest['branches'].update(branches())
    latest['branches']['stack'] = blend(latest['branches'], WEIGHTS, 1)
    latest['claim_check_enabled'] = True
    latest['claim_check'] = {'status': 'no_contradiction', 'source_window_id': 1, 'reason': 'Fixture'}
    state = {'session_id': 's', 'latest': latest, 'answer': {}}
    result = ShowcaseSession._result(state, 's')
    assert result['label'] == 'lie' and result['video_used_in_verdict']
    assert result['weights'] == WEIGHTS and result['manual_blend']
    latest['branches']['stack']['raw_score'] = .3
    assert ShowcaseSession._result(state, 's')['label'] == 'truth'
    latest['claim_check']['status'] = 'contradiction'
    assert ShowcaseSession._result(state, 's')['label'] == 'lie'
    latest['remote_video_enabled'] = False
    assert ShowcaseSession._result(state, 's')['label'] == 'inconclusive'


def test_runtime_blend_uses_all_same_answer_models_and_rejects_video_off(monkeypatch):
    runtime, video, dash, features, *_ = setup(monkeypatch)
    runtime.classifier_profile = 'webcam_pilot'; runtime.pilot_blend_weights = WEIGHTS
    source = branches()
    dash.infer = lambda *a, **k: {**copy.deepcopy(source['dasheng']), 'source_window_id': k['source_window_id'],
        '_pilot_decision': {**source['dasheng']['_pilot_decision'], 'source_window_id': k['source_window_id']}}
    video.predict = lambda *a, **k: {**source['vision'], 'source_window_id': k['window_id']}
    features.emotion = lambda *a: {k: v for k, v in copy.deepcopy(source['emotion']).items() if k != 'source_window_id'}
    clip, wave, text, ts = inputs()
    result = runtime.infer(clip, wave, text, 7, timestamps=ts)
    assert result['branches']['stack']['raw_score'] == pytest.approx(.55)
    assert result['branches']['emotion']['trained_classifier']['used_for_decision'] is True
    runtime.config['remote_vision'] = {'enabled': False}
    result = runtime.infer(clip, wave, text, 8, timestamps=ts)
    assert result['branches']['stack']['status'] == 'unavailable'
    runtime.close()


def test_weighted_video_requirement_blocks_capture_when_gpu_is_disabled():
    from types import SimpleNamespace
    from sunhacks.pipeline import Pipeline
    pipeline = Pipeline({'runtime_mode': 'split_local_video', 'remote_vision': {'enabled': False}})
    pipeline.runtime_preparation = {'status': 'ready', 'local_weights_loaded': True}
    pipeline.fast_runtime = SimpleNamespace(pilot_blend_weights=WEIGHTS, digest_stack=object(), snapshot=lambda: {})
    result = pipeline._split_runtime_state()
    assert result['live_ready'] is False
    assert 'requires GPU video' in result['reason']
