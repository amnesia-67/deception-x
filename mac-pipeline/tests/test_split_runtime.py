"""Split orchestration tests; no models, devices, sockets or participant input."""
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from sunhacks import split_runtime as module


class Component:
    def __init__(self):self.closed=False
    def prepare(self):return {"status":"ready","weights_loaded":True}
    def close(self):self.closed=True


class Dasheng(Component):
    def __init__(self,hook=None):super().__init__();self.hook=hook;self.calls=[]
    def infer(self,audio,**kwargs):
        if self.hook:self.hook()
        self.calls.append(kwargs["source_window_id"])
        return {"status":"trained","raw_score":.7,"per_fold_scores":[.7]*5}


class Features(Component):
    def __init__(self,emotion_hook=None):super().__init__();self.emotion_hook=emotion_hook;self.texts=[]
    def prepare(self):return {"components":{"text":{"status":"ready"},"emotion":{"status":"ready"}}}
    def text(self,text):
        self.texts.append(text)
        return {"status":"base","_full_embedding":np.ones(1024),"_intent_embeddings":np.ones((33,1024))}
    def emotion(self,audio):
        if self.emotion_hook:self.emotion_hook()
        return {"status":"base","_class_probabilities":np.full(9,1/9)}


class Faces(Component):
    def __init__(self,hook=None):super().__init__();self.hook=hook;self.received=[]
    def __call__(self,frames,timestamps):
        if self.hook:self.hook()
        self.received.append(np.array(timestamps))
        return frames,{"face_detect_rate":1,"timestamps":timestamps,"centre_crop_fallback":False}


class Stack(Component):
    def __init__(self):super().__init__();self.calls=[]
    def infer(self,row,dasheng_score,**kwargs):
        self.calls.append((dasheng_score,kwargs["source_window_id"]))
        return {"tabular":{"status":"trained_classifier","raw_score":.4},
                "stack":{"status":"trained_classifier","raw_score":.6,"not_used_modalities":["vision"]}}


class Video(Component):
    def __init__(self,hook=None):super().__init__();self.hook=hook;self.calls=[]
    def snapshot(self):return {"classifier_ready":True,"classifier_response_validated":False}
    def predict(self,clip,**kwargs):
        self.last_clip=clip
        self.calls.append(kwargs)
        if self.hook:self.hook()
        return {"status":"trained_classifier","model_id":"actual2.1","raw_score":.1,"classifier_only":True}


def inputs():
    return np.zeros((128,2,3,3),np.uint8),np.ones(16000,np.float32),{"status":"missing","text":""},np.arange(128)/30


def setup(monkeypatch,*,deadline=1,video_hook=None,asr_hook=None,emotion_hook=None,face_hook=None,dash_hook=None,transcript_status="base"):
    captured=[]
    def prepare_faces(frames,timestamps,preprocessor,*,already_selected=False):
        assert already_selected is True
        return preprocessor(frames,timestamps)
    monkeypatch.setattr(module,"prepare_digest_faces",prepare_faces)
    def row(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(missing_features=(),public_features=lambda:{"real_fixture":1})
    monkeypatch.setattr(module,"build_digest_row",row)
    video=Video(video_hook);dash=Dasheng(dash_hook);features=Features(emotion_hook);faces=Faces(face_hook);stack=Stack()
    events=[]
    def asr(audio,cfg):
        events.append("asr")
        if asr_hook:asr_hook()
        return {"status":transcript_status,"text":"Complete fixture answer"}
    runtime=module.SplitRuntime({"models":{"asr":{"path":"fixture"}},"split_runtime":{"deadline_seconds":deadline}},
        video_client=video,dasheng=dash,feature_models=features,face_preprocessor=faces,digest_stack=stack,
        prepare_asr_fn=lambda cfg:{"status":"ready"},transcribe_fn=asr,cleanup_asr_fn=lambda:events.append("asr_closed"))
    assert runtime.prepare()["local_weights_loaded"] is True
    return runtime,video,dash,features,faces,stack,captured,events


def test_all_independent_paths_overlap_and_semantics_stay_local(monkeypatch):
    lock=threading.Lock();seen=[];all_started=threading.Event();release=threading.Event()
    def hook(name):
        def enter():
            with lock:
                seen.append(name)
                if len(seen)==5:all_started.set()
            assert release.wait(1)
        return enter
    args=setup(monkeypatch,video_hook=hook("video"),asr_hook=hook("asr"),emotion_hook=hook("emotion"),face_hook=hook("face"),dash_hook=hook("dash"))
    runtime,video,dash,features,faces,stack,rows,events=args
    clip,audio,transcript,ts=inputs();result={}
    thread=threading.Thread(target=lambda:result.update(runtime.infer(clip,audio,transcript,7,timestamps=ts,duration_s=1)))
    thread.start()
    assert all_started.wait(1),seen
    assert features.texts==[],"Qwen ran before ASR completed"
    release.set();thread.join(2)
    assert not thread.is_alive()
    assert set(video.calls[0])=={"window_id","timestamps","face_crop"}
    assert video.calls[0]["window_id"]==7
    assert np.array_equal(video.calls[0]["timestamps"],ts)
    assert features.texts==["Complete fixture answer"]
    assert rows[0]["audio"] is audio
    assert rows[0]["duration_s"]==1
    assert stack.calls==[(.7,7)],"Video score entered the learned audio/digest stack"
    assert result["branches"]["stack"]["raw_score"]==.6
    assert result["locality"]["audio_transmitted"] is False
    assert all(branch["source_window_id"]==7 for branch in result["branches"].values())
    runtime.close()


def test_shared_timeout_never_reuses_prior_video_or_transcript(monkeypatch):
    release=threading.Event();entered=threading.Event()
    def slow():entered.set();release.wait(1)
    runtime,video,*_=setup(monkeypatch,deadline=.04,video_hook=slow)
    clip,audio,tr,ts=inputs()
    first=runtime.infer(clip,audio,tr,1,timestamps=ts)
    assert entered.is_set() and first["timings"]["jobs"]["vision"]["status"]=="timeout"
    second=runtime.infer(clip,audio,tr,2,timestamps=ts)
    assert second["timings"]["jobs"]["vision"]["status"]=="busy"
    assert second["branches"]["vision"]["raw_score"] is None
    assert len(video.calls)==1
    assert second["branches"]["stack"]["source_window_id"]==2
    release.set();runtime.close()


@pytest.mark.parametrize("status",["partial","missing","error"])
def test_incomplete_transcript_withholds_qwen_and_ensemble(monkeypatch,status):
    runtime,video,dash,features,faces,stack,rows,_=setup(monkeypatch,transcript_status=status)
    clip,audio,tr,ts=inputs();result=runtime.infer(clip,audio,tr,1,timestamps=ts)
    assert features.texts==[] and stack.calls==[] and rows==[]
    assert result["branches"]["stack"]["raw_score"] is None
    assert result["transcript"]["status"]==status
    runtime.close()


def test_close_defers_release_until_forward_finishes(monkeypatch):
    release=threading.Event();entered=threading.Event()
    def slow():entered.set();release.wait(1)
    runtime,video,dash,features,faces,stack,rows,events=setup(monkeypatch,deadline=.03,dash_hook=slow)
    clip,audio,tr,ts=inputs();runtime.infer(clip,audio,tr,1,timestamps=ts)
    assert entered.is_set()
    closed=runtime.close()
    assert closed["status"]=="closing" and dash.closed is False and features.closed is False
    release.set()
    deadline=time.monotonic()+1
    while not runtime.snapshot()["cleanup_done"] and time.monotonic()<deadline:time.sleep(.005)
    assert runtime.snapshot()["cleanup_done"] and all(c.closed for c in (dash,features,faces,stack))
    assert "asr_closed" in events


def test_duplicate_selected_times_are_preserved_without_second_selection(monkeypatch):
    runtime,video,dash,features,faces,stack,rows,_=setup(monkeypatch)
    clip,audio,tr,_=inputs();ts=np.repeat(np.arange(64)/30,2)
    result=runtime.infer(clip,audio,tr,3,timestamps=ts)
    assert np.array_equal(video.calls[0]["timestamps"],ts)
    assert np.array_equal(faces.received[0],ts)
    assert result["branches"]["stack"]["status"]=="trained_classifier"
    runtime.close()


def test_stop_during_asr_cancels_result_and_never_runs_qwen(monkeypatch):
    stopped=threading.Event()
    runtime,video,dash,features,faces,stack,rows,_=setup(monkeypatch,asr_hook=stopped.set)
    clip,audio,tr,ts=inputs();result=runtime.infer(clip,audio,tr,9,timestamps=ts,stop_event=stopped)
    assert features.texts==[] and stack.calls==[]
    assert all(v["status"]=="cancelled" for v in result["branches"].values())
    assert result["transcript"]["text"]=="" and result["features"]=={}
    runtime.close()


def test_missing_audio_cannot_fabricate_digest_even_if_video_runs(monkeypatch):
    runtime,video,dash,features,faces,stack,rows,_=setup(monkeypatch)
    clip,audio,tr,ts=inputs();result=runtime.infer(clip,None,tr,10,timestamps=ts)
    assert result["branches"]["vision"]["status"]=="trained_classifier"
    assert rows==[] and stack.calls==[]
    runtime.close()


def test_cleanup_attempts_all_components_and_exposes_error(monkeypatch):
    runtime,video,dash,features,faces,stack,rows,events=setup(monkeypatch)
    def fail():raise RuntimeError("cleanup fixture")
    dash.close=fail
    result=runtime.close()
    assert result["status"]=="cleanup_error"
    assert features.closed and faces.closed and stack.closed and "asr_closed" in events


def test_manual_session_context_cannot_silently_become_answer_transcript(monkeypatch):
    runtime,video,dash,features,faces,stack,rows,_=setup(monkeypatch)
    runtime.asr_config={"keep_loaded":True}
    clip,audio,_,ts=inputs()
    result=runtime.infer(clip,audio,{"status":"available","text":"Prior session text","scope":"session_context"},12,timestamps=ts)
    assert features.texts==[] and rows==[] and stack.calls==[]
    assert "session-context" in result["branches"]["text"]["reason"]
    runtime.close()


@pytest.mark.parametrize('component',['vision','dasheng','text','emotion','stack'])
def test_wrong_source_component_is_rejected_instead_of_relabelled(monkeypatch,component):
    runtime,video,dash,features,faces,stack,rows,_=setup(monkeypatch)
    target,method={'vision':(video,'predict'),'dasheng':(dash,'infer'),
                   'text':(features,'text'),'emotion':(features,'emotion'),'stack':(stack,'infer')}[component]
    original=getattr(target,method)
    def wrong_source(*args,**kwargs):
        value=original(*args,**kwargs)
        if component=='stack':value['stack']['source_window_id']=999
        else:value['source_window_id']=999
        return value
    setattr(target,method,wrong_source)
    clip,audio,tr,ts=inputs();result=runtime.infer(clip,audio,tr,7,timestamps=ts)
    assert result['branches'][component]['raw_score'] is None
    assert result['branches'][component]['status'] in ('error','unavailable')
    assert 'different source answer' in result['branches'][component]['reason']
    if component in ('dasheng','text','emotion'):
        assert stack.calls==[] and rows==[]
    runtime.close()


def test_output_guard_never_turns_stale_branch_into_current_success(monkeypatch):
    runtime,*_=setup(monkeypatch)
    out={'branches':{'vision':{'status':'trained_classifier','source_window_id':1,'raw_score':.9}},
         'features':{},'transcript':{},'timings':{}}
    result=runtime._finish_output(out,time.monotonic(),2,None)
    assert result['branches']['vision']['status']=='error'
    assert result['branches']['vision']['raw_score'] is None
    runtime.close()


@pytest.mark.parametrize('source',[True,np.bool_(True),1.,'1'])
def test_source_identity_requires_an_integer_not_coercible_alternative(source):
    with pytest.raises(ValueError,match='different source answer'):
        module.SplitRuntime._check_source({'source_window_id':source},1,'vision')
    module.SplitRuntime._check_source({'source_window_id':np.int64(1)},1,'vision')


def test_stale_asr_identity_is_rejected_before_qwen_and_not_relabelled(monkeypatch):
    runtime,video,dash,features,faces,stack,rows,_=setup(monkeypatch)
    runtime._transcribe=lambda audio,cfg:{'status':'base','text':'Old recognized words','source_window_id':999}
    clip,audio,tr,ts=inputs();result=runtime.infer(clip,audio,tr,7,timestamps=ts)
    assert features.texts==[] and stack.calls==[] and rows==[]
    assert result['branches']['text']['status']=='error'
    assert 'different source answer' in result['branches']['text']['reason']
    assert result['transcript']['text']==''
    runtime.close()


def test_stale_manual_identity_is_rejected_before_qwen_and_not_relabelled(monkeypatch):
    runtime,video,dash,features,faces,stack,rows,_=setup(monkeypatch)
    runtime.asr_config={'keep_loaded':True}
    clip,audio,_,ts=inputs()
    result=runtime.infer(clip,audio,{'status':'available','text':'Old manual words','scope':'answer','source_window_id':999},7,timestamps=ts)
    assert features.texts==[] and stack.calls==[] and rows==[]
    assert result['branches']['text']['status']=='error' and result['transcript']['text']==''
    runtime.close()


def test_timed_out_or_busy_asr_cannot_publish_prior_manual_text(monkeypatch):
    release=threading.Event()
    runtime,video,dash,features,faces,stack,rows,_=setup(monkeypatch,deadline=.03,asr_hook=lambda:release.wait(1))
    clip,audio,_,ts=inputs();manual={'status':'available','text':'Prior session context','scope':'session_context'}
    first=runtime.infer(clip,audio,manual,1,timestamps=ts)
    second=runtime.infer(clip,audio,manual,2,timestamps=ts)
    assert first['transcript']['text']=='' and second['transcript']['text']==''
    assert first['branches']['stack']['raw_score'] is None and second['branches']['stack']['raw_score'] is None
    release.set();runtime.close()
