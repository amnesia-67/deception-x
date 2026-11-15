import copy
import threading
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from sunhacks.showcase import ShowcaseSession
from sunhacks.temporal import AudioChunk


class Clock:
    def __init__(self): self.now = 100.0; self.hook = None
    def __call__(self): return self.now
    def wait(self, event, duration):
        self.now += max(duration, .1)
        if self.hook: self.hook()
        return event.is_set()


class Capture:
    def __init__(self, clock, *, silent=False, stalled=False, startup=False, stuck_stop=False, **kwargs):
        self.clock=clock; self.silent=silent; self.stalled=stalled; self.startup=startup
        self.stuck_stop=stuck_stop; self.started=False; self.stopped=False; self.frames=0
        self.options=kwargs
    def start(self): self.started=True
    def snapshot(self):
        status='stopping' if self.stopped and self.stuck_stop else 'stopped' if self.stopped else 'starting' if self.startup else 'running'
        return {'camera':{'status':status},'audio':{'status':status,'overflow_callbacks':0,'resampling_discontinuities':0}}
    def drain(self):
        if self.startup:return [],[]
        frame=[]
        if not self.stalled or not self.frames:
            frame=[SimpleNamespace(timestamp=self.clock(), sequence=self.frames, rgb=np.zeros((1,1,3),np.uint8))];self.frames+=1
        samples=np.full(1600,0 if self.silent else .1,np.float32)
        return frame,[AudioChunk(self.clock(),16000,samples)]
    def stop(self): self.stopped=True


def real_latest(score=.7):
    branches={key:{'status':status,'source_window_id':1} for key,status in {
        'vision':'trained_classifier','dasheng':'trained','text':'base','emotion':'base',
        'tabular':'trained_classifier','stack':'trained_classifier'}.items()}
    for key in ('vision','dasheng','tabular','stack'):branches[key]['raw_score']=score
    branches['stack'].update(threshold=.5,threshold_provenance='Reviewed legacy export serving rule.')
    branches['dasheng']['per_fold_scores']=[.2,.4,.5,.6,.8]
    branches['vision']['per_fold_scores']=[.3,.4,.5,.6,.7]
    return {'mock':False,'branches':branches,'window':{'id':1,'capture_mode':'whole_answer','duration_seconds':6.97,
            'audio':{'status':'available','coverage':1.,'missing_samples':0}},
            'features':{'audio':{'status':'ok','rms_dbfs':-24.,'peak_dbfs':-9.,'clipping_fraction':0.,
                        'activity_fraction_proxy':.5,'duration_seconds':6.97,'sample_rate':16000},
                        'video':{'status':'ok','brightness_mean':.5,'mean_absolute_frame_difference':.01,'frames':128}},
            'digest':{'face':{'status':'available','face_detect_rate':1.,'centre_crop_fallback':False}},
            'transcript':{'source':'local_asr','scope':'video_window','text':'A technical test sentence.'}}


class Pipeline:
    def __init__(self, clock):
        self.clock=clock;self.config={};self.session_id=None;self.started=None;self.stop_calls=0
        self.answer={};self.capture={'camera':{'status':'stopped'},'audio':{'status':'stopped'}}
        self.latest=None;self.start_options=None;self.auto_complete=True;self.model_ready=True
    def state(self):
        if self.started is not None and self.answer.get('status') not in ('stopped','error'):
            elapsed=self.clock()-self.started
            self.answer.update(elapsed_seconds=min(7.,elapsed),timer_started_at=self.started)
            if self.auto_complete and elapsed>=7:
                self.answer.update(status='complete',processing_ms=2000.)
                self.capture={'camera':{'status':'stopped'},'audio':{'status':'stopped'}}
                self.latest=real_latest()
        return copy.deepcopy({'session_id':self.session_id,'mode':'live' if self.started else 'stopped',
            'answer':self.answer,'capture':self.capture,'latest':self.latest,
            'runtime':{'live_ready':self.model_ready,'reason':'ready' if self.model_ready else 'not ready',
                       'lifecycle':{'active_jobs':[],'active_prepares':0}}})
    def start(self,mode,**kwargs):
        assert mode=='live';self.start_options=kwargs;self.started=self.clock();self.session_id='source-1'
        self.answer={'status':'capturing','elapsed_seconds':0.};self.capture={'camera':{'status':'running'},'audio':{'status':'running'}}
        return self.state()
    def stop(self):
        self.stop_calls+=1;self.answer['status']='stopped'
        self.capture={'camera':{'status':'stopped'},'audio':{'status':'stopped'}}
        return self.state()


def session(**options):
    clock=Clock();pipeline=Pipeline(clock);created=[]
    def factory(**kwargs):
        capture=Capture(clock,**options,**kwargs);created.append(capture);return capture
    controller=ShowcaseSession(pipeline,capture_factory=factory,clock=clock,wait=clock.wait,spawn=False)
    return controller,pipeline,clock,created


def run_warmup(controller):
    controller.start_warmup();controller._run_warmup(controller._generation,controller._event)
    return controller.snapshot()


def test_fifteen_second_practice_has_three_prompts_and_stops_sensors():
    controller,pipeline,clock,created=session();prompts=Counter()
    clock.hook=lambda:prompts.update([controller.snapshot()['prompt']['index']]) if controller.snapshot()['stage']=='warmup' else None
    value=run_warmup(controller)
    assert value['stage']=='ready' and value['ready'] and value['warmup']['completed']
    assert set(prompts)=={0,1,2} and all(45<=count<=55 for count in prompts.values())
    assert clock()==pytest.approx(115.,abs=.11)
    assert created[0].stopped and created[0].options['max_video_frames']==30
    assert value['warmup']['personal_calibration'] is False
    assert value['warmup']['content_verified'] is False
    assert 14.5<value['warmup']['observed_audio_seconds']<=15
    assert not pipeline.started


@pytest.mark.parametrize('options,reason',[
    ({'startup':True},'eight seconds'),({'stalled':True},'Camera stopped'),({'silent':True},'No microphone signal')])
def test_failed_warmup_stops_sensor_without_marking_ready(options,reason):
    controller,pipeline,clock,created=session(**options);value=run_warmup(controller)
    assert value['stage']=='error' and reason in value['error'] and not value['ready']
    assert created[0].stopped


def test_stuck_capture_blocks_ready_and_restart():
    controller,pipeline,clock,created=session(stuck_stop=True);value=run_warmup(controller)
    assert value['stage']=='error' and 'still stopping' in value['error']
    assert controller.is_active()
    with pytest.raises(RuntimeError):controller.start_warmup()


def test_warmup_requires_models_ready_and_idle_capture():
    controller,pipeline,clock,created=session();pipeline.model_ready=False
    with pytest.raises(RuntimeError,match='not ready'):controller.start_warmup()
    pipeline.model_ready=True;pipeline.capture['camera']['status']='running'
    with pytest.raises(RuntimeError,match='active recording'):controller.start_warmup()
    assert created==[]


def test_cancel_invalidates_old_warmup_before_a_new_run():
    controller,pipeline,clock,created=session();controller.start_warmup()
    generation,event=controller._generation,controller._event
    controller.cancel();controller.reset();controller.start_warmup()
    controller._run_warmup(generation,event)
    assert not created and controller.snapshot()['stage']=='starting'
    assert not controller._publish(generation,event,stage='ready')
    controller._run_warmup(controller._generation,controller._event)
    assert controller.snapshot()['stage']=='ready'


def test_cancel_during_warmup_stops_owned_capture_and_clears_ready():
    controller,pipeline,clock,created=session()
    def cancel_later():
        if clock()>102:controller.cancel()
    clock.hook=cancel_later
    value=run_warmup(controller)
    assert value['stage']=='cancelled' and not value['ready'] and created[0].stopped


def test_answer_uses_server_seven_second_timer_and_real_result():
    controller,pipeline,clock,created=session();run_warmup(controller)
    controller.start_answer();assert pipeline.start_options=={'auto_stop_seconds':7.0}
    controller._monitor_answer(controller._generation,controller._event)
    value=controller.snapshot()
    assert value['stage']=='result' and value['result']['label']=='lie'
    assert value['result']['score']==.7 and value['result']['calibrated'] is False
    assert value['result']['video_in_local_ensemble'] is False
    assert value['capture']['camera']['status']=='stopped'


def test_answer_can_start_directly_without_optional_practice():
    controller,pipeline,clock,created=session();clears=[]
    pipeline.clear_session_data=lambda:clears.append(True)
    controller.start_answer()
    assert pipeline.start_options=={'auto_stop_seconds':7.0} and clears==[True]
    controller._monitor_answer(controller._generation,controller._event)
    value=controller.snapshot()
    assert value['stage']=='result' and value['result']['practice_completed'] is False
    assert created==[]


def test_answer_during_optional_warmup_is_rejected():
    controller,pipeline,clock,created=session();controller.start_warmup()
    with pytest.raises(RuntimeError,match='active showcase'):controller.start_answer()
    assert pipeline.started is None


def test_cancel_answer_stops_only_the_matching_pipeline_session():
    controller,pipeline,clock,created=session();run_warmup(controller);controller.start_answer()
    generation,event=controller._generation,controller._event
    controller.cancel();assert pipeline.stop_calls==1
    controller._monitor_answer(generation,event)
    assert controller.snapshot()['stage']=='cancelled'
    assert controller.snapshot()['result'] is None


def test_cancel_old_answer_cannot_stop_new_dashboard_session():
    controller,pipeline,clock,created=session();run_warmup(controller);controller.start_answer()
    pipeline.session_id='another-session'
    controller.cancel();assert pipeline.stop_calls==0


def test_changed_source_session_discards_the_previous_answer():
    controller,pipeline,clock,created=session();run_warmup(controller);controller.start_answer()
    pipeline.session_id='another-session'
    controller._monitor_answer(controller._generation,controller._event)
    assert controller.snapshot()['stage']=='error'
    assert controller.snapshot()['result'] is None


def test_monitor_deadline_is_independent_of_browser_polling():
    controller,pipeline,clock,created=session();run_warmup(controller);pipeline.auto_complete=False;controller.start_answer()
    controller._monitor_answer(controller._generation,controller._event)
    assert controller.snapshot()['stage']=='error' and pipeline.stop_calls==1


@pytest.mark.parametrize('score,label',[(0.,'truth'),(.499,'truth'),(.5,'lie'),(1.,'lie')])
def test_pinned_half_score_rule_has_no_claimed_confidence(score,label):
    state={'session_id':'s','answer':{'processing_ms':2000},'latest':real_latest(score)}
    result=ShowcaseSession._result(state,'s')
    assert result['label']==label and result['score']==score and result['calibrated'] is False
    assert 'confidence' not in result


@pytest.mark.parametrize('mutate',[
    lambda s:s.update(session_id='old'),
    lambda s:s['latest'].update(mock=True),
    lambda s:s['latest']['branches']['stack'].update(status='unavailable'),
    lambda s:s['latest']['branches']['vision'].update(status='unavailable'),
    lambda s:s['latest']['branches']['text'].update(source_window_id=99),
    lambda s:s['latest']['branches']['stack'].update(raw_score=float('nan')),
    lambda s:s['latest']['branches']['stack'].update(raw_score=True),
    lambda s:s['latest']['window']['audio'].update(coverage=.99),
    lambda s:s['latest']['transcript'].update(text=''),
    lambda s:s['latest']['transcript'].update(scope='session_context'),
])
def test_unusable_or_stale_results_are_inconclusive(mutate):
    state={'session_id':'s','answer':{},'latest':real_latest()};mutate(state)
    result=ShowcaseSession._result(state,'s')
    assert result['label']=='inconclusive' and result['score'] is None


def test_close_never_allows_late_warmup_to_publish():
    controller,pipeline,clock,created=session();controller.start_warmup()
    generation,event=controller._generation,controller._event
    controller.close();controller._run_warmup(generation,event)
    assert controller.snapshot()['stage']=='cancelled'
    assert not controller.snapshot()['model_ready']
    with pytest.raises(RuntimeError,match='closed'):controller.start_warmup()


def test_cancel_cannot_stop_capture_before_its_nonblocking_start_finishes():
    controller,pipeline,clock,created=session()
    entered=threading.Event();release=threading.Event();stopped=threading.Event();order=[]
    class StartBoundaryCapture(Capture):
        def start(self):
            order.append('start_enter');entered.set()
            assert release.wait(2)
            super().start();order.append('start_finished')
        def stop(self):
            order.append('stop');super().stop();stopped.set()
    cap=StartBoundaryCapture(clock)
    controller._capture_factory=lambda **kw:cap
    controller.start_warmup()
    generation,event=controller._generation,controller._event
    worker=threading.Thread(target=controller._run_warmup,args=(generation,event))
    worker.start();assert entered.wait(1)
    canceller=threading.Thread(target=controller.cancel);canceller.start()
    assert not stopped.wait(.05)
    release.set();worker.join(2);canceller.join(2)
    assert not worker.is_alive() and not canceller.is_alive()
    assert order.index('start_finished')<order.index('stop')
    assert controller.snapshot()['stage']=='cancelled' and cap.stopped


def test_new_participant_warmup_and_reset_clear_prior_pipeline_data():
    controller,pipeline,clock,created=session();clears=[]
    pipeline.latest=real_latest()
    def clear():
        clears.append(True);pipeline.latest=None
    pipeline.clear_session_data=clear
    controller.start_warmup()
    assert len(clears)==1 and pipeline.latest is None
    controller._run_warmup(controller._generation,controller._event)
    pipeline.latest=real_latest()
    controller.reset()
    assert len(clears)==2 and pipeline.latest is None
    assert controller.snapshot()['stage']=='idle'


def test_prepare_models_reconnects_idle_pipeline_without_opening_sensors():
    controller,pipeline,clock,created=session();pipeline.model_ready=False;calls=[]
    assert controller.snapshot()['can_prepare'] is True
    assert controller.snapshot()['preparing'] is False
    pipeline.prepare_runtime=lambda:calls.append('prepare')
    controller.prepare_models()
    assert calls==['prepare'] and created==[]


@pytest.mark.parametrize('busy_kind',['capture','answer','jobs','prepares','thread','status','showcase'])
def test_prepare_is_disabled_and_rejected_during_capture_or_model_work(busy_kind):
    controller,pipeline,clock,created=session();pipeline.model_ready=False
    original_state=pipeline.state
    def state():
        value=original_state()
        if busy_kind=='capture':value['capture']['camera']['status']='running'
        if busy_kind=='answer':value['answer']['status']='processing'
        if busy_kind=='jobs':value['runtime']['lifecycle']['active_jobs']=['vision']
        if busy_kind=='prepares':value['runtime']['lifecycle']['active_prepares']=1
        if busy_kind=='status':value['runtime']['status']='preparing'
        return value
    pipeline.state=state
    if busy_kind=='thread':pipeline.prepare_thread=SimpleNamespace(is_alive=lambda:True)
    if busy_kind=='showcase':controller._state['stage']='warmup'
    pipeline.prepare_runtime=lambda:pytest.fail('Preparation launched while busy')
    value=controller.snapshot()
    assert value['can_prepare'] is False
    assert value['preparing'] is (busy_kind in ['prepares','thread','status'])
    with pytest.raises(RuntimeError):controller.prepare_models()
    assert created==[]


def test_cancelled_processing_can_prepare_after_outstanding_jobs_finish():
    controller,pipeline,clock,created=session();run_warmup(controller);controller.start_answer()
    controller.cancel();pipeline.model_ready=False
    assert controller.snapshot()['stage']=='cancelled'
    assert controller.snapshot()['can_prepare'] is True
    calls=[];pipeline.prepare_runtime=lambda:calls.append(True)
    controller.prepare_models();assert calls==[True]


@pytest.mark.parametrize('stage',['cancelled','error'])
def test_answer_can_restart_directly_from_cancelled_or_error_after_clearing_data(stage):
    controller,pipeline,clock,created=session();controller._state['stage']=stage;clears=[]
    pipeline.clear_session_data=lambda:clears.append(True)
    controller.start_answer()
    assert pipeline.start_options=={'auto_stop_seconds':7.0} and clears==[True]


def test_completed_optional_practice_is_reported_with_the_result():
    controller,pipeline,clock,created=session();run_warmup(controller);controller.start_answer()
    controller._monitor_answer(controller._generation,controller._event)
    assert controller.snapshot()['result']['practice_completed'] is True


def test_ready_does_not_require_optional_practice():
    controller,pipeline,clock,created=session()
    assert controller.snapshot()['ready'] is True
    assert controller.snapshot()['warmup']['completed'] is False
    pipeline.capture['camera']['status']='running'
    assert controller.snapshot()['ready'] is False


@pytest.mark.parametrize('problem',['no_face','fallback','missing_features','nan_audio','nan_video','nan_face',
    'digital_silence','all_clipped','missing_classifier_score','invalid_classifier_score'])
def test_bad_input_quality_cannot_become_a_binary_verdict(problem):
    latest=real_latest();state={'session_id':'s','answer':{},'latest':latest}
    if problem=='no_face':latest['digest']['face']['face_detect_rate']=0
    elif problem=='fallback':latest['digest']['face']['centre_crop_fallback']=True
    elif problem=='missing_features':latest.pop('features')
    elif problem=='nan_audio':latest['features']['audio']['rms_dbfs']=float('nan')
    elif problem=='nan_video':latest['features']['video']['brightness_mean']=float('nan')
    elif problem=='nan_face':latest['digest']['face']['face_detect_rate']=float('nan')
    elif problem=='digital_silence':latest['features']['audio'].update(rms_dbfs=-160.,peak_dbfs=-160.)
    elif problem=='all_clipped':latest['features']['audio']['clipping_fraction']=1.
    elif problem=='missing_classifier_score':latest['branches']['vision'].pop('raw_score')
    elif problem=='invalid_classifier_score':latest['branches']['dasheng']['raw_score']=float('inf')
    result=ShowcaseSession._result(state,'s')
    assert result['label']=='inconclusive' and result['score'] is None


def test_model_evidence_reports_raw_scores_without_confidence_or_new_threshold():
    latest=real_latest(.499999);latest['branches']['stack'].update(model_id='verified-export',
        cv_metrics={'accuracy':.60},calibration_scope='exported source domain',caveats=['Different capture domain'])
    state={'session_id':'s','answer':{},'latest':latest}
    result=ShowcaseSession._result(state,'s')
    assert result['label']=='truth' and result['threshold']==.5
    assert result['distance_from_threshold']==pytest.approx(.000001)
    assert result['branch_scores']=={'ensemble':.499999,'dasheng':.499999,'digest':.499999,'video':.499999}
    assert result['video_used_in_verdict'] is False
    assert result['fold_score_ranges']['dasheng']=={'min':.2,'max':.8,'count':5}
    assert result['validation_context']['reported_cv_metrics']=={'accuracy':.60}
    assert 'confidence' not in result


def test_latest_healthy_input_example_keeps_the_trained_serving_rule():
    latest=real_latest(.3771)
    latest['branches']['dasheng']['raw_score']=.3101
    latest['branches']['tabular']['raw_score']=.3783
    latest['branches']['vision']['raw_score']=.4823
    latest['features']['audio'].update(rms_dbfs=-30.16,peak_dbfs=-4.65,activity_fraction_proxy=.244)
    latest['digest']['face']['face_detect_rate']=.992
    result=ShowcaseSession._result({'session_id':'s','answer':{},'latest':latest},'s')
    assert result['label']=='truth' and result['score']==.3771
    assert result['branch_scores']['video']==.4823


def test_recognized_text_is_only_disclosed_for_a_matching_valid_answer():
    state={'session_id':'s','answer':{},'latest':real_latest()}
    result=ShowcaseSession._result(state,'s')
    assert result['recognized_text']=='A technical test sentence.'
    state['latest']['branches']['text']['source_window_id']=999
    assert 'recognized_text' not in ShowcaseSession._result(state,'s')
    state['latest']['branches']['text']['source_window_id']=1
    state['latest']['features']['audio'].update(rms_dbfs=-160.,peak_dbfs=-160.)
    assert 'recognized_text' not in ShowcaseSession._result(state,'s')


def test_rejected_source_never_exposes_participant_scores_or_words():
    state={'session_id':'wrong','answer':{},'latest':real_latest(.9)}
    value=ShowcaseSession._result(state,'expected')
    assert value['label']=='inconclusive'
    assert all(v is None for v in value['branch_scores'].values())
    assert value['fold_score_ranges']=={}
    assert 'recognized_text' not in value


@pytest.mark.parametrize('score,label',[(.45,'lie'),(.4,'truth'),(.4238521751969923,'lie')])
def test_reviewed_export_threshold_controls_model_label(score,label):
    latest=real_latest(score)
    latest['branches']['stack'].update(threshold=.4238521751969923,
        threshold_provenance='Prior-matched rule fitted on training CV.')
    result=ShowcaseSession._result({'session_id':'s','answer':{},'latest':latest},'s')
    assert result['label']==label and result['model_label']==label
    assert result['threshold']==.4238521751969923
    assert result['distance_from_threshold']==pytest.approx(abs(score-.4238521751969923))
    assert result['threshold_provenance']=='Prior-matched rule fitted on training CV.'


@pytest.mark.parametrize('threshold',[None,True,False,'0.424',float('nan'),float('inf'),-.1,1.1])
def test_invalid_export_threshold_withholds_verdict_scores_and_words(threshold):
    latest=real_latest(.45)
    latest['branches']['stack']['threshold']=threshold
    result=ShowcaseSession._result({'session_id':'s','answer':{},'latest':latest},'s')
    assert result['label']=='inconclusive' and result['model_result_valid'] is False
    assert result['score'] is None and result['threshold'] is None
    assert all(value is None for value in result['branch_scores'].values())
    assert 'recognized_text' not in result


def test_claim_check_is_a_server_setting_that_defaults_on():
    controller,pipeline,clock,created=session()
    assert controller.claim_check_enabled() is True
    assert controller.snapshot()['claim_check_enabled'] is True
    assert controller.configure_claim_check(False)['claim_check_enabled'] is False
    assert controller.claim_check_enabled() is False
    for value in ('false',None,0):
        with pytest.raises(ValueError):controller.configure_claim_check(value)
    assert controller.claim_check_enabled() is False
