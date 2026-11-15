"""Pinned final 850-clip export: exact fusion and diagnostic emotion LoRA."""
import hashlib,io,math,threading,time
from pathlib import Path
import torch
from torch import nn
from .local_dasheng import (_json_bytes,LocalDashengUnavailable,AttnPoolHead,
    tensor_schema,validate_fold_state,load_fold_state,interp_time,eval_audio)

FUSION_SHA='891b32e0ac840f37d4a89a91b76ce893842055bc44b832767ced846bf0b36080'
EMOTION_MANIFEST_SHA='13f6839008396f774f852b60705bb7d45d63ad2d905b4b587e95813ac8705886'
EMOTION_FOLD_SHA='45ad793c95695b72ef1d50c39a6f651c62eabceb26c656feabb7f847739b77d1'

def sigmoid(x):
    return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))

def read_fusion(path,audio_manifest_sha):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=FUSION_SHA:raise LocalDashengUnavailable('Final pilot fusion hash mismatch')
    f=_json_bytes(raw)
    if f.get('format')!='sunhacks-pilot-webcam-fusion-v1' or f.get('profile')!='webcam_pilot' or set(f.get('branches',{}))!={'dasheng'}:
        raise LocalDashengUnavailable('Unsupported final pilot fusion contract')
    b=f['branches']['dasheng']
    if b['manifest_sha256']!=audio_manifest_sha:raise LocalDashengUnavailable('Fusion/audio manifest mismatch')
    vals=[b['logit_mean'],b['logit_sd'],f['calibration']['A'],f['calibration']['B'],f['threshold']]
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in vals) or b['logit_sd']<=0 or not 0<f['threshold']<1:
        raise LocalDashengUnavailable('Invalid final pilot fusion parameters')
    return f

def fuse(logit,f):
    if isinstance(logit,bool) or not isinstance(logit,(int,float)) or not math.isfinite(logit):raise ValueError('Finite audio logit required')
    b=f['branches']['dasheng'];z=(logit-b['logit_mean'])/b['logit_sd']
    return {'fused_logit':z,'probability':sigmoid(f['calibration']['A']*z+f['calibration']['B']),
            'threshold':f['threshold'],'fusion_sha256':FUSION_SHA}

class _EmotionBackbone(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,wav):
        src=torch.nn.functional.layer_norm(wav.float(),wav.shape[-1:])
        return interp_time(self.model.extract_features(src,padding_mask=None)['x'])

class _EmotionFold(nn.Module):
    def __init__(self,model,config):
        super().__init__();self.backbone=_EmotionBackbone(model);self.head=AttnPoolHead(1024,dropout=config['head_dropout'])

class PilotEmotion:
    """Share immutable BF16-rounded base; disable LoRA for the original digest.

    The owner holds FrozenFeatureModels._lock across both forwards. Adapter
    toggles therefore cannot change another answer's features. No second base
    is loaded, and the trained classifier never supplies digest probabilities.
    """
    def __init__(self,folder,base_model):
        from peft import LoraConfig,inject_adapter_in_model
        p=Path(folder);raw=(p/'deploy_meta.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=EMOTION_MANIFEST_SHA:raise LocalDashengUnavailable('Pilot emotion manifest hash mismatch')
        self.meta=_json_bytes(raw)
        if self.meta.get('backbone')!='emotion2vec' or self.meta.get('profile')!='webcam_pilot' or self.meta.get('model')!='emotion2vec_lora_webcam_pilot':raise LocalDashengUnavailable('Wrong pilot emotion profile')
        data=(p/'fold0.pt').read_bytes();entry=self.meta['folds']['fold0.pt']
        if hashlib.sha256(data).hexdigest()!=EMOTION_FOLD_SHA or entry['sha256']!=EMOTION_FOLD_SHA or len(data)!=entry['size']:raise LocalDashengUnavailable('Pilot emotion checkpoint mismatch')
        ck=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True);cfg=ck['config']
        if cfg!=entry['config'] or cfg.get('backbone')!='emotion2vec' or tensor_schema(ck['state'])!=entry['state_schema']:raise LocalDashengUnavailable('Pilot emotion state/config mismatch')
        n=len(base_model.blocks);idx='|'.join(str(i) for i in range(n-cfg['lora_blocks'],n))
        spec=LoraConfig(r=cfg['lora_r'],lora_alpha=2*cfg['lora_r'],lora_dropout=cfg['lora_dropout'],target_modules=rf'blocks\.({idx})\.attn\.(qkv|proj)',bias='none')
        base_model=inject_adapter_in_model(spec,base_model)
        self.model=_EmotionFold(base_model,cfg).float().cpu().eval().requires_grad_(False)
        validate_fold_state(self.model,ck['state']);load_fold_state(self.model,ck['state'])
        self.adapters(False)

    def adapters(self,enabled):
        from peft.tuners.tuners_utils import BaseTunerLayer
        for m in self.model.backbone.model.modules():
            if isinstance(m,BaseTunerLayer):m.enable_adapters(enabled)

    def infer(self,audio):
        start=time.perf_counter();a=eval_audio(audio);self.adapters(True)
        try:
            with torch.inference_mode():logit=float(self.model.head(self.model.backbone(torch.from_numpy(a)[None])).float())
        finally:self.adapters(False)
        if not math.isfinite(logit):raise ValueError('Nonfinite emotion logit')
        raw=sigmoid(logit);c=self.meta['calibration'];p=sigmoid(c['A']*math.log(min(1-1e-6,max(1e-6,raw))/(1-min(1-1e-6,max(1e-6,raw))))+c['B'])
        return {'status':'trained','model_id':'emotion2vec_lora_webcam_pilot','profile':'webcam_pilot',
                'manifest_sha256':EMOTION_MANIFEST_SHA,'logit':logit,'raw_score':raw,
                'calibrated_probability':p,'threshold':self.meta['threshold'],'used_for_decision':False,
                'role':'diagnostic_only','locality':'mac','elapsed_ms':round((time.perf_counter()-start)*1000,3)}
