"""Technical local-only benchmark; existing non-person speech fixture, no sensors/network."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO','0.55')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO','0.45')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
# This script deliberately does not load the manager env file.
import json
import argparse
import sys
import time
import threading
from pathlib import Path
import numpy as np
import psutil
import torch
from scipy.io import wavfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sunhacks.fast_runtime import FastRuntime
from sunhacks.pipeline import public,peak_rss_mb

parser=argparse.ArgumentParser(description="Measure local resident-model paths only; no capture, network, manager or task heads")
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--audio',type=Path,required=True,help='Existing permitted non-person mono16kHz WAV technical fixture')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--repeats',type=int,default=5)
args=parser.parse_args()
if not 1 <= args.repeats <= 20:parser.error('--repeats must be within1..20')
config=json.loads(args.config.read_text())
if config.get('runtime_mode')!='remote_fast':parser.error('Requires remote_fast profile')
torch.set_num_threads(config['cpu_threads'])
torch.set_num_interop_threads(1)
process=psutil.Process()
samples=[]
monitor_stop=threading.Event()
def monitor():
    while not monitor_stop.wait(.05):samples.append(process.memory_info().rss/2**20)
threading.Thread(target=monitor,daemon=True).start()
def memory():
    return {'rss_mb':round(process.memory_info().rss/2**20,3),
            'lifetime_peak_rss_mb':round(peak_rss_mb(),3),
            'mps_current_mb':round(torch.mps.current_allocated_memory()/2**20,3),
            'mps_driver_mb':round(torch.mps.driver_allocated_memory()/2**20,3)}
rate,raw=wavfile.read(args.audio)
assert rate==16000 and raw.ndim==1
wave=raw.astype(np.float32)/(np.iinfo(raw.dtype).max+1) if np.issubdtype(raw.dtype,np.integer) else raw.astype(np.float32)
wave=np.ascontiguousarray(wave[:63000])
assert len(wave)==63000
clip=np.zeros((64,96,128,3),np.uint8)
runtime=FastRuntime(config)
report={'benchmark_type':'local_required_paths_only_remote_offline',
        'input':'Existing generated non-person speech fixture; 3.9375 seconds at16kHz. In-memory color clip is not encoded because remote vision is off.',
        'models':{name:{k:v for k,v in config['models'][name].items() if k in ('revision','device','dtype','keep_loaded')} for name in ('asr','dasheng','text','emotion')},
        'cpu_threads':config['cpu_threads'],'remote_called':False,'sensors_opened':False,'manager_called':False,
        'old_fusion_called':False,'memory_before':memory(),'trials':[]}
try:
    report['preparation']=runtime.prepare()
    report['memory_prepared']=memory()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(public(report),indent=2)+'\n')
    print(json.dumps({'preparation_status':report['preparation']['status'],'local_weights_loaded':report['preparation'].get('local_weights_loaded'),'elapsed_ms':report['preparation'].get('elapsed_ms'),'memory':report['memory_prepared']}),flush=True)
    for identifier in range(1,args.repeats+1):
        output=runtime.infer(clip,wave,{'text':'','status':'missing','source':'manual'},identifier)
        item={'id':identifier,'timings':output['timings'],'branches':{name:{k:v for k,v in branch.items() if k in ('status','elapsed_ms','embedding_shape','load_cache_hit','device','model_id','reason')} for name,branch in output['branches'].items() if name in ('vision','dasheng','text','emotion')},
              'asr':{k:v for k,v in output['transcript'].items() if k not in ('text','segments')},
              'transcript_character_count':len(output['transcript'].get('text','')),'memory':memory()}
        future=output.get('_emotion_future')
        if future:
            supplemental=future.result(timeout=30)
            item['emotion_completed_after_required_paths']={k:v for k,v in supplemental.items() if not k.startswith('_')}
        report['trials'].append(item)
        print(json.dumps({'trial':identifier,'service_ms':output['timings']['service_ms'],'asr_status':output['transcript'].get('status'),'text_status':output['branches']['text']['status'],'dasheng_status':output['branches']['dasheng']['status'],'memory':item['memory']}),flush=True)
        args.output.write_text(json.dumps(public(report),indent=2)+'\n')
    report['local_required_service_ms']=[item['timings']['service_ms'] for item in report['trials']]
    report['memory_after']=memory()
finally:
    report['shutdown']=runtime.close()
    monitor_stop.set()
    report['sampled_peak_rss_mb']=round(max(samples,default=0),3)
    report['limitations']=['No remote vision, replacement fusion or live capture measured. These are local technical timings, not full pipeline throughput or predictive-quality validation.','Repeated fixed speech fixture; sample count and input diversity do not establish production p95.','Preparation loads weights only; first-forward compilation is measured in trial1.','Emotion runs independently; wait for completion outside required-path timing only between benchmark trials.']
    args.output.write_text(json.dumps(public(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'saved':str(args.output),'shutdown':report['shutdown'],'sampled_peak_rss_mb':report['sampled_peak_rss_mb']}),flush=True)
