"""CPU-only fake-backbone deployment checks; no weights, CUDA, datasets or sockets."""
import copy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import jepa_deploy as deploy
from src import jepa_service as service

spec = importlib.util.spec_from_file_location('export_jepa21_test', ROOT / 'scripts/export_deploy.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

@dataclass
class FakeConfig:
    backbone: str = 'vjepa2_1_l'
    lora_r: int = 2
    lora_blocks: int = 1
    lora_dropout: float = 0.1
    head_dropout: float = 0.3

class FakeBackbone(nn.Module):
    name, n_frames, res, out_dim = 'vjepa2_1_l', 64, 384, 1024
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(2,1024))
        self.lora_B = nn.Parameter(torch.zeros(1024,2))
        self.frozen = nn.Parameter(torch.zeros(1),requires_grad=False)
    def forward(self,x):
        assert x.shape[1] == 64
        return torch.ones(x.shape[0],32,1024)

class FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(1024)
        self.score=nn.Linear(1024,1)
        self.out=nn.Linear(1024,1)
    def forward(self,x): return self.out(self.norm(x)).mean(1).squeeze(-1)

class FakeModel(nn.Module):
    def __init__(self,cfg=None,device='cpu'):
        super().__init__()
        self.backbone=FakeBackbone()
        self.head=FakeHead()


def state():
    model=FakeModel()
    return {'lora':{k:v.clone() for k,v in model.backbone.state_dict().items() if 'lora_' in k},
            'head':{k:v.clone() for k,v in model.head.state_dict().items()}}


def training_files(tmp_path):
    source=tmp_path/'runs/tune_vjepa2_1_l'
    source.mkdir(parents=True)
    base=tmp_path/deploy.BASE_RELATIVE_PATH
    base.parent.mkdir(parents=True)
    base.write_bytes(b'fake base encoder for CPU tests only')
    config=vars(FakeConfig())
    for fold in range(5):
        torch.save({'config':dict(config),'state':state(),'info':{'must_not_export':'training-only'},
                    'test_ids':['private-id']},source/f'fold{fold}.pt')
    (source/'summary.json').write_text(json.dumps({'protocol':'synthetic-test-only',
        'oof':{'auroc':0.5,'acc':0.5},'auroc_ci95':[0.1,0.9],
        'folds':[{'test':{'auroc':0.5}}]*5}))
    return source


def bundle(tmp_path,monkeypatch):
    source=training_files(tmp_path)
    monkeypatch.setattr(exporter,'ROOT',str(tmp_path))
    monkeypatch.setattr(service,'ROOT',str(tmp_path))
    monkeypatch.setitem(sys.modules,'src.tuning',types.SimpleNamespace(TuneConfig=FakeConfig,TunedModel=FakeModel))
    def select_frames(n_total,n_out,start): return start+np.arange(n_out)*(64//n_out)
    monkeypatch.setitem(sys.modules,'src.clip_store',types.SimpleNamespace(select_frames=select_frames,eval_starts=lambda n:[0,n-64]))
    return exporter.export_jepa(),source


def test_no_train_export_and_strict_service_load(tmp_path,monkeypatch):
    target,_=bundle(tmp_path,monkeypatch)
    assert 'scripts.train_stage2' not in sys.modules
    meta=json.loads((target/'deploy_meta.json').read_text())
    assert meta['contract']['hidden_width']==1024
    assert meta['contract']['frames_per_view']==64
    assert set(meta['folds'])=={f'fold{k}.pt' for k in range(5)}
    for name,entry in meta['folds'].items():
        ck,sha,size=deploy.read_checkpoint(target/name)
        assert set(ck)=={'config','state'}
        assert entry['sha256']==sha and entry['size']==size
    loaded=service.JEPAService(str(target),device='cpu')
    assert loaded.info()['model']=='vjepa2_1_vitl_lora_5fold'
    assert loaded.info()['model_id'].endswith('vit_large_384')
    assert loaded.info()['contract']['classifier_only'] is True
    assert len(loaded.folds)==5


def test_two_cache_views_select64_stride1(tmp_path,monkeypatch):
    target,_=bundle(tmp_path,monkeypatch)
    loaded=service.JEPAService(str(target),device='cpu')
    crops=np.arange(128,dtype=np.uint8).reshape(128,1,1,1).repeat(3,-1)
    windows=loaded._windows(crops)
    assert [win.shape[0] for win in windows]==[64,64]
    assert windows[0][:,0,0,0].tolist()==list(range(0,64))
    assert windows[1][:,0,0,0].tolist()==list(range(64,128))


def test_fake_classifier_predict_uses_L_geometry_without_CUDA(tmp_path,monkeypatch):
    target,_=bundle(tmp_path,monkeypatch)
    loaded=service.JEPAService(str(target),device='cpu')
    monkeypatch.setattr(loaded,'_prepare',lambda payload:(np.zeros((128,1,1,3),dtype=np.uint8),None))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:pytest.fail('CPU fake must not touch CUDA'))
    result=loaded.predict({'frames':['synthetic']*128})
    assert result['model']==deploy.MODEL_LABEL
    assert result['classifier_only'] is True
    assert result['calibrated'] is False and 'verdict' not in result
    assert result['raw_score']==result['p_neural']
    assert len(result['per_fold_probability'])==5
    assert len(result['temporal_attention_32'])==32
    assert '_embedding' not in result and 'sequence' not in result


@pytest.mark.parametrize('mutation',[
    lambda st:st['lora'].pop('lora_A'),
    lambda st:st['lora'].__setitem__('unknown',torch.ones(1)),
    lambda st:st['lora'].__setitem__('lora_A',torch.ones(1)),
    lambda st:st['lora'].__setitem__('lora_A',torch.ones(2,1024,dtype=torch.float64)),
    lambda st:st['head'].pop('norm.weight'),
    lambda st:st['head'].__setitem__('norm.weight',torch.full((1024,),float('nan'))),
])
def test_bad_fold_never_partially_mutates_shared_model(mutation):
    model=FakeModel()
    original=copy.deepcopy(model.state_dict())
    bad=state()
    bad['lora']['lora_B'].fill_(123)
    mutation(bad)
    with pytest.raises(ValueError):deploy.load_fold_state_strict(model,bad)
    for key,value in model.state_dict().items():assert torch.equal(value,original[key])


def test_complete_fold_replaces_every_adapter_and_head():
    model=FakeModel()
    replacement=state()
    replacement['lora']['lora_A'].fill_(42)
    replacement['head']['norm.weight'].fill_(3)
    deploy.load_fold_state_strict(model,replacement)
    assert torch.all(model.backbone.lora_A==42)
    assert torch.all(model.head.norm.weight==3)


def test_hash_checked_before_restricted_load(tmp_path,monkeypatch):
    path=tmp_path/'fold.pt'
    path.write_bytes(b'changed bytes')
    monkeypatch.setattr(torch,'load',lambda *args,**kwargs:pytest.fail('Digest mismatch must fail before deserialize'))
    with pytest.raises(ValueError,match='SHA256'):deploy.read_checkpoint(path,expected_sha256='a'*64)


def test_restricted_loader_flag_is_explicit(tmp_path,monkeypatch):
    path=tmp_path/'fold.pt'
    path.write_bytes(b'mocked payload')
    calls=[]
    def load(*args,**kwargs):
        calls.append(kwargs)
        return {'config':vars(FakeConfig()),'state':state()}
    monkeypatch.setattr(torch,'load',load)
    deploy.read_checkpoint(path)
    assert calls==[{'map_location':'cpu','weights_only':True}]


@pytest.mark.parametrize('change',['wrong-family','extra-fold','missing-fold','hash','partial-state','base-hash'])
def test_service_refuses_mismatched_deployment(tmp_path,monkeypatch,change):
    target,_=bundle(tmp_path,monkeypatch)
    meta_path=target/'deploy_meta.json'
    meta=json.loads(meta_path.read_text())
    if change=='wrong-family':meta['contract']['backbone']='vjepa2'
    elif change=='extra-fold':(target/'fold5.pt').write_bytes(b'extra')
    elif change=='missing-fold':(target/'fold4.pt').unlink()
    elif change=='hash':meta['folds']['fold0.pt']['sha256']='0'*64
    elif change=='base-hash':meta['base_encoder']['sha256']='0'*64
    else:
        ck,_,_=deploy.read_checkpoint(target/'fold0.pt')
        ck['state']['lora'].pop('lora_A')
        torch.save(ck,target/'fold0.pt')
        _,digest,size=deploy.read_checkpoint(target/'fold0.pt')
        meta['folds']['fold0.pt'].update(sha256=digest,size=size,state_schema=deploy.state_schema(ck['state']))
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):service.JEPAService(str(target),device='cpu')


def test_existing_export_directory_is_never_overwritten(tmp_path,monkeypatch):
    target,_=bundle(tmp_path,monkeypatch)
    with pytest.raises(FileExistsError):exporter.export_jepa()
    assert (target/'deploy_meta.json').exists()


def test_nonfinite_fake_forward_fails_instead_of_emitting_verdict(tmp_path,monkeypatch):
    target,_=bundle(tmp_path,monkeypatch)
    loaded=service.JEPAService(str(target),device='cpu')
    monkeypatch.setattr(loaded,'_prepare',lambda payload:(np.zeros((128,1,1,3),dtype=np.uint8),None))
    for model in loaded.models.values():
        monkeypatch.setattr(model.backbone,'forward',lambda x:torch.full((1,32,1024),float('nan')))
    with pytest.raises(ValueError,match='nonfinite'):loaded.predict({'frames':['synthetic']*128})


@pytest.mark.parametrize('family',['vjepa2_1','vjepa2'])
def test_historical_B_and_vjepa2_folds_are_rejected(tmp_path,monkeypatch,family):
    path=tmp_path/'fold.pt'
    path.write_bytes(b'mocked payload')
    monkeypatch.setattr(torch,'load',lambda *a,**k:{'config':{**vars(FakeConfig()),'backbone':family},'state':state()})
    with pytest.raises(ValueError,match='only V-JEPA 2.1 ViT-L'):deploy.read_checkpoint(path)


def test_contract_is_vit_large():
    c=deploy.CONTRACT
    assert c['frames_per_view']==64 and c['native_temporal_tubelets']==32
    assert c['variant']=='ViT-L' and c['hidden_width']==1024 and c['encoder']['depth']==24 and c['encoder']['heads']==16
    assert deploy.MODEL_ID.endswith('vjepa2_1_vit_large_384') and 'vitl' in deploy.BASE_RELATIVE_PATH
