"""Reviewed single-model webcam-pilot Dasheng export, CPU float32 inference."""
from __future__ import annotations
import hashlib
import io
import math
from pathlib import Path
import threading
import torch
from .local_dasheng import (TunedDashengEnsemble,LocalDashengUnavailable,_json_bytes,CONFIG_FIELDS,
    tensor_schema,validate_fold_state,load_fold_state,reviewed_base_factory)

MANIFEST_SHA256='5381b13f06d70cf4ad062f23998c50429eee03cf46ac43ab4b6e9b8b58c41cd2'
FOLD_SHA256='c7577bcd3b9eaaf4904c58d21035adb5ca909375f6a50f89a74441747f4f7afa'
SOURCE_REVISION='143e924'
PROFILE='webcam_pilot'
MODEL='dasheng_lora_webcam_pilot'


def calibrated_probability(raw,calibration):
    if isinstance(raw,bool) or not isinstance(raw,(int,float)) or not math.isfinite(raw) or not 0<=raw<=1:
        raise LocalDashengUnavailable('Invalid pilot raw probability')
    values=[calibration.get(k) for k in ('A','B')]
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in values):
        raise LocalDashengUnavailable('Invalid pilot calibration')
    clipped=min(1-1e-6,max(1e-6,raw));x=values[0]*math.log(clipped/(1-clipped))+values[1]
    return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))


class PilotDasheng(TunedDashengEnsemble):
    def __init__(self,manifest_path,fold_dir,*,base_path=None,source_path=None,model_factory=None,fusion_path=None):
        raw=Path(manifest_path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA256:raise LocalDashengUnavailable('Pilot manifest differs from the reviewed export')
        self.manifest_sha256=MANIFEST_SHA256;self.meta=_json_bytes(raw)
        from .pilot_final import read_fusion
        self.fusion=read_fusion(fusion_path,MANIFEST_SHA256) if fusion_path else None
        if any(self.meta.get(k)!=v for k,v in {'format':'sunhacks-pilot-webcam-v1','profile':PROFILE,'backbone':'dasheng','model':MODEL}.items()):
            raise LocalDashengUnavailable('Wrong pilot model/profile')
        entries=self.meta.get('folds');folder=Path(fold_dir)
        if not isinstance(entries,dict) or set(entries)!={'fold0.pt'} or {p.name for p in folder.glob('*.pt')}!={'fold0.pt'}:
            raise LocalDashengUnavailable('Pilot audio export requires exactly one model, fold0.pt')
        entry=entries['fold0.pt'];data=(folder/'fold0.pt').read_bytes()
        if not 0<len(data)<=4*1024*1024 or len(data)!=entry.get('size') or hashlib.sha256(data).hexdigest()!=entry.get('sha256') or entry['sha256']!=FOLD_SHA256:
            raise LocalDashengUnavailable('Pilot checkpoint hash/size mismatch')
        ck=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
        if (not isinstance(ck,dict) or set(ck)!={'config','state'} or not isinstance(ck['config'],dict)
                or set(ck['config'])-CONFIG_FIELDS or ck['config']!=entry.get('config')
                or ck['config'].get('backbone')!='dasheng' or tensor_schema(ck['state'])!=entry.get('state_schema')):
            raise LocalDashengUnavailable('Pilot checkpoint config or state schema differs')
        threshold=self.meta.get('threshold')
        if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<threshold<1:
            raise LocalDashengUnavailable('Pilot threshold is invalid')
        calibrated_probability(.5,self.meta.get('calibration',{}))
        if model_factory is None:model_factory=reviewed_base_factory(base_path,source_path)
        model=model_factory(ck['config']).float().cpu().eval()
        for param in model.parameters():param.requires_grad_(False)
        validate_fold_state(model,ck['state']);load_fold_state(model,ck['state'])
        self.models={'pilot':model};self.folds=[('fold0.pt','pilot',ck['state'])]
        self.policy={'source_revision':SOURCE_REVISION};self.meta['class_1_meaning']='deceptive'
        self._closed=False;self._lock=threading.Lock()

    def prepare(self):
        return {**super().prepare(),'classifier':MODEL,'profile':PROFILE,'folds':1}

    def infer(self,*args,**kwargs):
        out=super().infer(*args,**kwargs)
        p=calibrated_probability(out['raw_score'],self.meta['calibration'])
        out.update(classifier=MODEL,profile=PROFILE,score_aggregation='single_model_sigmoid_logit',
            feature_aggregation='single_model_temporal_features',logit=out['per_fold_logits'][0],calibrated_probability=p)
        out['_pilot_decision']={'status':'trained_classifier','classifier_only':True,'profile':PROFILE,'model_id':MODEL,
            'source_window_id':out['source_window_id'],'raw_score':p,'uncalibrated_audio_score':out['raw_score'],
            'logit':out['logit'],'threshold':float(self.meta['threshold']),
            'threshold_provenance':self.meta['threshold_provenance'],'calibrated':False,
            'calibration_scope':'Platt fit on 120 webcam-pilot validation clips; used for selection; calibration reliability unestablished',
            'calibration':dict(self.meta['calibration']),'used_modalities':['tuned_dasheng_webcam_pilot'],
            'not_used_modalities':['vision','xgb_digest','qwen','emotion'],'weights':{MODEL:1.},'locality':'mac',
            'cv_metrics':self.meta['metrics'],'caveats':self.meta['caveats'],'manifest_sha256':self.manifest_sha256,
            'elapsed_ms':out['elapsed_ms'],'score_semantics':'experimental_pilot_score_not_honesty_probability'}
        if self.fusion is not None:
            from .pilot_final import fuse
            f=fuse(out['logit'],self.fusion)
            out['_pilot_decision'].update(raw_score=f['probability'],threshold=f['threshold'],
                fusion_sha256=f['fusion_sha256'],fused_logit=f['fused_logit'],
                classifier='webcam_pilot_final_dasheng',used_modalities=['audio'])
        return out
