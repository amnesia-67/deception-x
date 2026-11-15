import threading
from sunhacks.claim_runtime import ClaimRuntime
from test_showcase import real_latest
from sunhacks.showcase import ShowcaseSession

class Checker:
    deadline=.2
    def __init__(self,**changes):self.changes=changes
    def check(self,text,**kwargs):
        return {'source_window_id':kwargs['source_window_id'],'status':'contradiction','kind':'multimodal','reason':'Generated fixture decision.','evidence':[],
            'call_made':True,'sent_keyframes':2,'sent_transcript':True,'provider':'openrouter','model':'nace-ai/drex-max-20261016',
            'upstream_provider':'DeepInfra','requested_model':'nace-ai/drex-max','model_version_verified':False,
            'decision_choice':'contradiction','decision_probabilities':{'contradiction':.98,'no_contradiction':.01,'insufficient_evidence':.01},**self.changes}

def check(checker):
    r=ClaimRuntime({},checker=checker)
    try:
        t=r.begin(images=('generated','generated'),visual_authorized=True,source_window_id=4,stop_event=threading.Event())
        t.on_transcript({'status':'base','source':'local_asr','scope':'video_window','source_window_id':4,'text':'Generated statement.'})
        return t.finish()
    finally:r.close()

def test_same_answer_multimodal_decision_and_diagnostics_survive_dispatch():
    v=check(Checker());assert v['status']=='contradiction' and v['sent_transcript'] is True
    assert v['source_window_id']==4 and v['decision_probabilities']['contradiction']==.98
    assert v['upstream_provider']=='DeepInfra' and v['model_version_verified'] is False

def test_missing_transmitted_source_transcript_cannot_assert_contradiction():
    assert check(Checker(sent_transcript=False))['status']=='unverifiable'

def test_timeout_accounting_does_not_claim_text_or_images_stayed_local():
    v=check(Checker(status='unverifiable',sent_keyframes=None,sent_transcript=None,attempted_keyframes=2,attempted_transcript=True,
        transmission_status='attempted_outcome_unknown',call_outcome_unknown=True))
    assert v['status']=='unverifiable' and v['call_outcome_unknown'] is True and v['sent_transcript'] is None

def test_insufficient_evidence_leaves_trained_prediction_unchanged():
    for score,label in [(.3,'truth'),(.7,'lie')]:
        latest=real_latest(score);latest['claim_check_enabled']=True
        latest['claim_check']={'source_window_id':1,'status':'unverifiable','kind':'multimodal','reason':'Insufficient evidence.','evidence':[]}
        result=ShowcaseSession._result({'session_id':'s','answer':{},'latest':latest},'s')
        assert result['label']==label and result['model_label']==label
