"""Native-feature contracts using tiny CPU stubs, no weights or model downloads."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from sunhacks.frozen_features import FrozenFeatureModels,eval_model_audio
from sunhacks.local_digest import INTENTS,format_testimony,EMOTIONS


class Tokenizer:
    def __init__(self):self.calls=[];self.bad_padding=False
    def __call__(self,texts,**kwargs):
        self.calls.append((texts,kwargs))
        attention=torch.ones(len(texts),2,dtype=torch.int64)
        if self.bad_padding:attention[:,-1]=0
        return {"input_ids":torch.ones(len(texts),2,dtype=torch.int64),"attention_mask":attention}


class Qwen(torch.nn.Module):
    def forward(self,**inputs):
        n=inputs["input_ids"].shape[0]
        hidden=torch.zeros(n,2,1024);hidden[:,0]=999
        hidden[:,-1]=torch.arange(1,1025)
        return SimpleNamespace(last_hidden_state=hidden)


class Emotion(torch.nn.Module):
    def __init__(self):
        super().__init__();self.inputs=[];self.proj=torch.nn.Linear(1024,9,bias=False)
        with torch.no_grad():
            self.proj.weight.zero_();self.proj.weight[:,0]=torch.arange(9)/2
    def extract_features(self,waveform,padding_mask=None):
        self.inputs.append(waveform.clone())
        return {"x":torch.stack((torch.ones(1,1024),torch.ones(1,1024)*3),1)}


class Registry:
    def __init__(self):self.tok=Tokenizer();self.qwen=Qwen();self.emotion=Emotion();self.closed=False
    def _load(self,name):
        return (self.qwen,self.tok,"cpu",torch.float32,False) if name=="text" else (SimpleNamespace(model=self.emotion),None,"cpu",torch.float32,False)
    def close(self):self.closed=True


def test_raw_intents_are_prepared_once_and_testimony_uses_exact_prefix():
    reg=Registry();models=FrozenFeatureModels({},registry=reg)
    prepared=models.prepare()
    assert prepared["status"]=="ready"
    documents=[text for texts,opts in reg.tok.calls for text in texts]
    assert documents==list(INTENTS)
    assert all(opts["max_length"]==64 for _,opts in reg.tok.calls)
    assert models.intent_embeddings.shape==(33,1024)
    assert np.allclose(np.linalg.norm(models.intent_embeddings,axis=1),1)
    count=len(reg.tok.calls);models.prepare();assert len(reg.tok.calls)==count
    output=models.text("Fixture words")
    assert reg.tok.calls[-1][0]==[format_testimony("Fixture words")]
    assert reg.tok.calls[-1][1]["max_length"]==512
    assert np.array_equal(output["_full_embedding"],np.arange(1,1025))
    assert output["_embedding"].shape==(128,) and np.linalg.norm(output["_embedding"])==pytest.approx(1)
    assert output["tuning"]=="frozen_base_for_digest"
    models.close();assert reg.closed


def test_emotion_uses_real_projection_of_mean_sequence_not_embedding_as_probs():
    reg=Registry();models=FrozenFeatureModels({},registry=reg)
    output=models.emotion(np.arange(8000,dtype=np.float32)/8000)
    assert reg.emotion.inputs[0].shape==(1,16000)
    assert float(reg.emotion.inputs[0].mean())==pytest.approx(0,abs=1e-5)
    expected=torch.softmax(torch.arange(9,dtype=torch.float32),dim=0).numpy()
    assert np.allclose(output["_class_probabilities"],expected)
    assert list(output["class_probabilities"])==list(EMOTIONS)
    assert np.array_equal(output["_embedding"],np.full(1024,2))
    assert output["model_audio_samples"]==16000
    models.close()


def test_qwen_rejects_wrong_padding_before_forward():
    reg=Registry();models=FrozenFeatureModels({},registry=reg);models.prepare()
    reg.tok.bad_padding=True
    with pytest.raises(ValueError,match="left-padded"):
        models.text("Fixture")
    models.close()


def test_digest_frozen_contract_rejects_tuned_encoder_configuration():
    with pytest.raises(ValueError,match="frozen"):
        FrozenFeatureModels({"models":{"text":{"adapter_path":"trained"}}})


def test_eval_model_audio_center_crop_and_observed_short_audio_padding_only():
    source=np.arange(70*16000,dtype=np.float32)
    assert np.array_equal(eval_model_audio(source),source[5*16000:65*16000])
    with pytest.raises(ValueError):eval_model_audio(None)
    with pytest.raises(ValueError):eval_model_audio(np.zeros(7999,np.float32))
