"""Fetch the reviewed small deployment exports without checking out the GPU tree.

Run from this Mac's Python environment. Base encoders remain separately cached;
this command never fetches recordings/datasets or runs training/capture.
"""
from __future__ import annotations
import argparse,hashlib,io,ipaddress,json,shutil,subprocess,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REF='a7cad71f019894461a7fb17edd027fd9f7007c3d'
PINS={
 'dasheng/deploy_meta.json':'5381b13f06d70cf4ad062f23998c50429eee03cf46ac43ab4b6e9b8b58c41cd2',
 'dasheng/fold0.pt':'c7577bcd3b9eaaf4904c58d21035adb5ca909375f6a50f89a74441747f4f7afa',
 'emotion2vec/deploy_meta.json':'13f6839008396f774f852b60705bb7d45d63ad2d905b4b587e95813ac8705886',
 'emotion2vec/fold0.pt':'45ad793c95695b72ef1d50c39a6f651c62eabceb26c656feabb7f847739b77d1',
 'vjepa2_1_l/deploy_meta.json':'e0910ae6d979521db7f2af25a91d22647c0be87a462cd0ff5af9f686a5fdb9f8',
 'vjepa2_1_l/fold0.pt':'f85aa149b6b6613a37b6fc531a198cd0247b981cf630a2d5f1fef2741cdf49a9',
 'fusion.json':'891b32e0ac840f37d4a89a91b76ce893842055bc44b832767ced846bf0b36080',
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.split-answer.local.json')
    parser.add_argument('--gpu-host',help='Optional numeric LAN or Tailscale IP; stays in ignored local config')
    parser.add_argument('--gpu-port',type=int,default=5002)
    args=parser.parse_args()
    if args.gpu_host:ipaddress.ip_address(args.gpu_host)
    if not 1<=args.gpu_port<=65535:parser.error('Port must be within 1..65535')
    if not args.config.name.endswith('.local.json'):parser.error('Use an ignored *.local.json configuration')
    config=json.loads((args.config if args.config.exists() else ROOT/'config.split-answer.example.json').read_text())
    git=['git']
    if shutil.which('gh'):git+=['-c','credential.helper=','-c','credential.helper=!gh auth git-credential']
    subprocess.run(git+['fetch','origin','main'],cwd=ROOT.parent,check=True)
    data=subprocess.check_output(git+['archive',REF,'models/pilot_webcam','models/dasheng_lora',
        'models/branch2_xgboost.json','models/branch2_digest_schema.json',
        'fixtures/golden_v1/pilot_webcam_expected.json','fixtures/golden_v1/audio_f32le_16k.bin'],cwd=ROOT.parent)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for name,expected in PINS.items():
            raw=archive.extractfile('models/pilot_webcam/'+name).read()
            if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Reviewed export hash mismatch: '+name)
        destination=ROOT/'.models'/('deployment-'+REF);destination.mkdir(parents=True,exist_ok=True)
        archive.extractall(destination,filter='data')
    folder=destination/'models/pilot_webcam'
    config['classifier_profile']='webcam_pilot'
    config['pilot_audio']={'manifest_path':str(folder/'dasheng/deploy_meta.json'),'fold_dir':str(folder/'dasheng'),
        'manifest_sha256':PINS['dasheng/deploy_meta.json'],'source_revision':'143e924','fusion_path':str(folder/'fusion.json')}
    config['pilot_emotion']={'fold_dir':str(folder/'emotion2vec')}
    config.setdefault('local_ensemble',{}).update(xgb_path=str(destination/'models/branch2_xgboost.json'),fold_dir=str(destination/'models/dasheng_lora'))
    remote=config.setdefault('remote_vision',{})
    remote.update(profile='webcam_pilot',expected_manifest_sha256=PINS['vjepa2_1_l/deploy_meta.json'],advertised_video_model='vjepa2_1_vitl_lora_webcam_pilot')
    if args.gpu_host:
        remote.update(host=args.gpu_host,port=args.gpu_port,enabled=True)
        if ipaddress.ip_address(args.gpu_host) in ipaddress.ip_network('100.64.0.0/10'):remote['connection_type']='tailscale'
    config.setdefault('deployment',{}).update(classifier_profile='webcam_pilot',local_export_revision='143e924',
        model_branch_revision=REF,reported_vision='V-JEPA 2.1 ViT-L webcam-pilot',
        artifact_reason='Final 850-clip pilot: exported Dasheng decision; emotion2vec and ViT-L are diagnostic. Heart rate is display only.')
    args.config.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.config.with_suffix(args.config.suffix+'.tmp')
    temporary.write_text(json.dumps(config,indent=2)+'\n');temporary.replace(args.config)
    print('Verified final pilot exports installed. Local config updated; base encoders and credentials remain unchanged.')

if __name__=='__main__':main()
