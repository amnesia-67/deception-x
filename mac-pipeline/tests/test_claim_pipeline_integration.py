import copy
import threading
import time
from types import SimpleNamespace
import pytest
from sunhacks.showcase import ShowcaseSession
from test_answer_capture import queued_answer
from test_showcase import real_latest


class Task:
    def __init__(self,status='contradiction'):
        self.calls=[];self.cancelled=False;self.status=status
    def on_transcript(self,value):self.calls.append(copy.deepcopy(value))
    def finish(self,transcriptfallback=None):
        return {'status':self.status,'kind':'math','source_window_id':1,'reason':'The parsed equation is incorrect.' if self.status=='contradiction' else 'No arithmetic contradiction found; not proof of truth.','evidence':['1 + 1 = 3; calculated 2.'],'call_made':False,'sent_keyframes':0}
    def cancel(self):self.cancelled=True


class Claims:
    def __init__(self,task):self.task=task;self.begins=[]
    def begin(self,**kwargs):self.begins.append(kwargs);return self.task


def run_answer(monkeypatch,*,visual=False,fail=False):
    p=queued_answer();p.config['claim_check']={'enabled':True}
    p.answer.update(visual_claim_check=visual)
    task=Task();claims=Claims(task);p.claim_runtime=claims
    latest=real_latest(.3)
    class Runtime:
        def infer(self,clip,audio,transcript,window_id,**kwargs):
            if fail:raise RuntimeError('fixture model failure')
            recognized={'status':'base','source':'local_asr','scope':'video_window','source_window_id':window_id,'text':'One plus one equals three.'}
            kwargs['on_transcript'](recognized)
            return {'branches':latest['branches'],'transcript':recognized,'features':latest['digest'],'timings':{'service_ms':1},'locality':{'audio':'mac'}}
    p.fast_runtime=Runtime()
    return p,task,claims


def test_guard_runs_from_same_asr_and_never_selects_images_without_optin(monkeypatch):
    p,task,claims=run_answer(monkeypatch)
    p._infer_answer()
    assert p.answer['status']=='complete'
    assert claims.begins[0]['images']==() and claims.begins[0]['visual_authorized'] is False
    assert len(task.calls)==1 and task.calls[0]['source_window_id']==1
    assert p.latest['claim_check']['status']=='contradiction'
    assert p.latest['model_processing_ms']>=0
    value=ShowcaseSession._result({"session_id":p.session_id,"answer":p.answer,"latest":p.latest},p.session_id)
    assert value['label']=='lie' and value['decision_source']=='supported_contradiction' and value['model_label']=='truth'
    assert value['model_result_valid'] is True


def test_only_two_frames_selected_after_explicit_visual_optin(monkeypatch):
    p,task,claims=run_answer(monkeypatch,visual=True)
    p._infer_answer()
    assert claims.begins[0]['visual_authorized'] is True
    assert len(claims.begins[0]['images'])==2


def test_failed_model_cancels_owned_claim_task_without_publishing(monkeypatch):
    p,task,claims=run_answer(monkeypatch,fail=True)
    p._infer_answer()
    assert p.latest is None and task.cancelled and p.answer['status']=='error'






@pytest.mark.parametrize('status,expected',[('contradiction','lie'),('no_contradiction','truth'),('unverifiable','truth'),('not_applicable','truth')])
def test_guard_cannot_confirm_truth_or_erase_a_contradiction(status,expected):
    state={'session_id':'s','answer':{},'latest':real_latest(.3)}
    state['latest']['claim_check_enabled']=True
    state['latest']['claim_check']={'status':status,'kind':'math','source_window_id':1,'reason':'fixture evidence','evidence':[]}
    r=ShowcaseSession._result(state,'s')
    assert r['label']==expected and r['model_label']=='truth'
    assert r['model_result_valid'] is True


def test_stale_guard_cannot_override_current_classifier():
    state={'session_id':'s','answer':{},'latest':real_latest(.3)}
    state['latest']['claim_check_enabled']=True
    state['latest']['claim_check']={'status':'contradiction','kind':'math','source_window_id':999,'reason':'stale','evidence':['private prior evidence']}
    r=ShowcaseSession._result(state,'s')
    assert r['label']=='inconclusive' and r['claim_check']['evidence']==[]


def test_no_contradiction_cannot_upgrade_a_lie_prediction():
    value={'label':'lie','model_label':'lie','source_window_id':1}
    latest={'claim_check':{'status':'no_contradiction','kind':'math','source_window_id':1,'reason':'not proof of truth','evidence':[]}}
    assert ShowcaseSession._guarded_verdict(value,latest)['label']=='lie'


def test_showcase_choices_reach_pipeline_without_implicit_image_consent():
    from test_showcase import session
    controller,pipeline,clock,created=session()
    value=controller.start_answer(visual_claim_check=True)
    assert pipeline.start_options=={'auto_stop_seconds':7.0,'visual_claim_check':True}
    assert value['visual_claim_check'] is True




@pytest.mark.parametrize('option',[{'visual_claim_check':'true'},{'visual_claim_check':None}])
def test_invalid_option_never_creates_capture(option):
    from test_showcase import session
    controller,pipeline,clock,created=session()
    with pytest.raises(ValueError):controller.start_answer(**option)
    assert pipeline.start_options is None
