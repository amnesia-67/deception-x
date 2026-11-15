"""Transcript guard-dispatch ordering and identity; no models, APIs or sensors."""
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from test_split_runtime import setup,inputs


@pytest.mark.parametrize('status',['base','available'])
def test_valid_same_answer_asr_dispatches_once_before_qwen_without_awaiting_result(monkeypatch,status):
    runtime,video,dash,features,faces,stack,rows,_=setup(monkeypatch,transcript_status=status)
    order=[];received=[]
    original=features.text
    def text(value):order.append('qwen');return original(value)
    features.text=text
    class UnfinishedFuture:
        def result(self,*args,**kwargs):pytest.fail('Transcript callback future was awaited')
    def dispatch(value):
        order.append('hook');received.append(value)
        return UnfinishedFuture()
    clip,audio,tr,ts=inputs()
    result=runtime.infer(clip,audio,tr,7,timestamps=ts,on_transcript=dispatch)
    assert order==['hook','qwen'] and len(received)==1
    assert received[0]['source_window_id']==7 and received[0]['source']=='local_asr'
    assert received[0]['scope']=='video_window'
    assert result['claim_hook_dispatched'] is True and 'claim_hook_error' not in result
    assert result['branches']['stack']['status']=='trained_classifier'
    runtime.close()


@pytest.mark.parametrize('status,text',[('partial','Some words'),('missing',''),('error','Untrusted words'),('base',''),('available','   ')])
def test_partial_failed_or_empty_asr_never_dispatches(monkeypatch,status,text):
    runtime,_,_,features,_,stack,_,_=setup(monkeypatch)
    runtime._transcribe=lambda audio,cfg:{'status':status,'text':text}
    called=[];clip,audio,tr,ts=inputs()
    result=runtime.infer(clip,audio,tr,7,timestamps=ts,on_transcript=called.append)
    assert called==[] and features.texts==[] and stack.calls==[]
    assert 'claim_hook_dispatched' not in result and 'claim_hook_error' not in result
    runtime.close()


@pytest.mark.parametrize('source',[999,True,np.bool_(True),'7',7.0])
def test_invalid_asr_source_is_rejected_before_dispatch(monkeypatch,source):
    runtime,_,_,features,_,stack,_,_=setup(monkeypatch)
    runtime._transcribe=lambda audio,cfg:{'status':'base','text':'Old recognized words','source_window_id':source}
    called=[];clip,audio,tr,ts=inputs()
    result=runtime.infer(clip,audio,tr,7,timestamps=ts,on_transcript=called.append)
    assert called==[] and features.texts==[] and stack.calls==[]
    assert result['transcript']['text']==''
    assert result['branches']['text']['status']=='error'
    assert 'different source answer' in result['branches']['text']['reason']
    runtime.close()


def test_callback_exception_is_diagnostic_and_preserves_good_audio_text_and_stack(monkeypatch):
    runtime,_,_,features,_,stack,_,_=setup(monkeypatch)
    called=[]
    def fail(value):
        called.append(value)
        raise RuntimeError('private fixture diagnostic should not be emitted')
    clip,audio,tr,ts=inputs();result=runtime.infer(clip,audio,tr,7,timestamps=ts,on_transcript=fail)
    assert len(called)==1 and features.texts==['Complete fixture answer'] and stack.calls==[(.7,7)]
    assert result['branches']['text']['status']=='base'
    assert result['branches']['stack']['status']=='trained_classifier'
    assert result['claim_hook_dispatched'] is False
    assert result['claim_hook_error']=={'source_window_id':7,'error_type':'RuntimeError','reason':'Transcript guard dispatch failed.'}
    assert 'private fixture' not in str(result)
    runtime.close()


def test_callback_gets_a_copy_and_cannot_change_qwen_words_or_source_identity(monkeypatch):
    runtime,_,_,features,_,_,_,_=setup(monkeypatch)
    def mutate(value):value.update(text='Changed words',source_window_id=999)
    clip,audio,tr,ts=inputs();result=runtime.infer(clip,audio,tr,7,timestamps=ts,on_transcript=mutate)
    assert features.texts==['Complete fixture answer']
    assert result['transcript']['text']=='Complete fixture answer'
    assert result['transcript']['source_window_id']==7
    runtime.close()


def test_manual_answer_text_does_not_invoke_the_asr_only_hook(monkeypatch):
    runtime,_,_,features,_,_,_,_=setup(monkeypatch)
    runtime.asr_config={'keep_loaded':True}
    called=[];clip,audio,_,ts=inputs()
    result=runtime.infer(clip,audio,{'status':'available','text':'Same answer manual text','scope':'answer','source_window_id':7},7,
                         timestamps=ts,on_transcript=called.append)
    assert called==[] and features.texts==['Same answer manual text']
    assert 'claim_hook_dispatched' not in result
    runtime.close()


def test_stop_after_decoder_prevents_dispatch_and_qwen(monkeypatch):
    stopped=threading.Event()
    runtime,_,_,features,_,_,_,_=setup(monkeypatch,asr_hook=stopped.set)
    called=[];clip,audio,tr,ts=inputs()
    result=runtime.infer(clip,audio,tr,7,timestamps=ts,on_transcript=called.append,stop_event=stopped)
    assert called==[] and features.texts==[] and result['transcript']['text']==''
    runtime.close()


def test_no_callback_keeps_the_existing_output_contract(monkeypatch):
    runtime,*_=setup(monkeypatch)
    clip,audio,tr,ts=inputs();result=runtime.infer(clip,audio,tr,7,timestamps=ts)
    assert result['transcript']['text']=='Complete fixture answer'
    assert result['branches']['stack']['status']=='trained_classifier'
    assert 'claim_hook_dispatched' not in result and 'claim_hook_error' not in result
    runtime.close()
