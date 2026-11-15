"""Local, participant-grouped collection; no model decisions or dataset uploads."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tempfile
import uuid

import numpy as np

SPLITS=('train','validation','test')
LABELS=('truth','lie')
PARTICIPANT=re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')


def write_json(path, value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


class DatasetStore:
    def __init__(self,root):
        self.root=Path(root).expanduser().resolve()

    def initialize(self):
        self.root.mkdir(parents=True,exist_ok=True)
        for split in SPLITS:
            for label in LABELS:(self.root/split/label).mkdir(parents=True,exist_ok=True)
        (self.root/'.pending').mkdir(exist_ok=True)
        path=self.root/'dataset.json'
        if not path.exists():
            write_json(path,{'schema':'sunhacks-labeled-av-v1','seed':secrets.token_hex(32),
                'split_ratios':{'train':.75,'validation':.15,'test':.10},
                'split_unit':'participant','label_source':'participant_self_report',
                'participants':{}})
        value=json.loads(path.read_text())
        if value.get('schema')!='sunhacks-labeled-av-v1':raise RuntimeError('Unsupported dataset manifest')
        return value

    def split_for(self,participant,manifest):
        if not isinstance(participant,str) or not PARTICIPANT.fullmatch(participant):
            raise ValueError('Use a participant code of 1–64 letters, numbers, underscores or hyphens')
        existing=manifest['participants'].get(participant)
        if existing is not None:
            if existing not in SPLITS:raise RuntimeError('Invalid participant split in dataset manifest')
            return existing
        # Stable seeded random participant assignment. Labels and model outputs
        # never affect assignment. All answers from a participant share a split.
        number=int(hashlib.sha256((manifest['seed']+'\0'+participant).encode()).hexdigest(),16)/2**256
        return 'train' if number<.75 else 'validation' if number<.90 else 'test'

    def counts(self):
        return {split:{label:sum(1 for path in (self.root/split/label).glob('*/metadata.json') if path.is_file())
                       for label in LABELS} for split in SPLITS}

    def encode(self,frames,audio,start,devices,*,cancelled):
        """Keep every captured frame with original timestamps in a sidecar.

        MP4 is a VFR H.264/AAC review copy; audio.wav is the canonical lossless
        float32 16kHz mono signal. Encoding performs no face crops or inference.
        """
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:raise RuntimeError('ffmpeg is required to save recordings')
        self.initialize();take=uuid.uuid4().hex
        pending=self.root/'.pending'/take;pending.mkdir()
        try:
            from PIL import Image
            from scipy.io import wavfile
            wavfile.write(pending/'audio.wav',16000,np.asarray(audio,dtype=np.float32))
            with tempfile.TemporaryDirectory(prefix='frames-',dir=pending) as work:
                work=Path(work);lines=['ffconcat version 1.0']
                for i,frame in enumerate(frames):
                    if cancelled.is_set():raise RuntimeError('Recording cancelled')
                    Image.fromarray(frame.rgb).save(work/f'{i:04d}.png',compress_level=1)
                    end=frames[i+1].timestamp-start if i+1<len(frames) else 7.
                    lines += [f"file '{i:04d}.png'",'option framerate 1000',f'duration {end-(frame.timestamp-start):.9f}']
                lines += [f"file '{len(frames)-1:04d}.png'",'option framerate 1000']
                (work/'frames.ffconcat').write_text('\n'.join(lines)+'\n')
                command=[ffmpeg,'-nostdin','-v','error','-y','-f','concat','-safe','0','-i',str(work/'frames.ffconcat'),
                    '-i',str(pending/'audio.wav'),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','18',
                    '-pix_fmt','yuv420p','-fps_mode','vfr','-enc_time_base','1:90000','-video_track_timescale','90000',
                    '-c:a','aac','-b:a','128k','-t','7','-movflags','+faststart',str(pending/'video.mp4')]
                process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                try:
                    import time
                    deadline=time.monotonic()+45
                    while process.poll() is None:
                        if cancelled.wait(.05):raise RuntimeError('Recording cancelled')
                        if time.monotonic()>deadline:raise RuntimeError('Video encoding exceeded 45 seconds')
                    error=process.stderr.read(2048)
                    if process.returncode:raise RuntimeError('Video encoding failed; recording was not saved')
                finally:
                    if process.poll() is None:process.kill()
                    process.wait();process.stderr.close()
            if cancelled.is_set():raise RuntimeError('Recording cancelled')
            stamps=[float(frame.timestamp-start) for frame in frames]
            from datetime import datetime,timezone,timedelta
            import time
            captured=(datetime.now(timezone.utc)-timedelta(seconds=time.monotonic()-start)).isoformat()
            metadata={'captured_utc':captured,'schema':'sunhacks-labeled-av-v1','take_id':take,'duration_seconds':7.,
                'frame_timestamps_seconds':stamps,'frame_sequences':[int(f.sequence) for f in frames],
                'source_frame_count':len(frames),'observed_capture_fps':(len(frames)-1)/(stamps[-1]-stamps[0]),
                'max_frame_gap_seconds':float(np.max(np.diff(stamps))),
                'video':{'file':'video.mp4','encoding':'H.264 CRF18, VFR; no face crop','width':int(frames[0].rgb.shape[1]),
                         'height':int(frames[0].rgb.shape[0]),'timestamp_source':'host_monotonic_after_camera_read',
                         'hardware_av_sync':False,'timestamp_origin':'first retained camera frame'},
                'audio':{'file':'audio.wav','sample_rate':16000,'channels':1,'dtype':'float32','samples':int(len(audio)),
                         'coverage':1.,'rms':float(np.sqrt(np.mean(np.square(audio,dtype=np.float64)))),
                         'clipped_fraction':float(np.mean(np.abs(audio)>=.999)),'mp4_audio':'AAC review copy; use audio.wav for training'},
                'devices':devices,'label_source':'participant_self_report','model_calls':0,
                'files_sha256':{name:hashlib.sha256((pending/name).read_bytes()).hexdigest() for name in ('video.mp4','audio.wav')}}
            write_json(pending/'metadata.json',metadata)
            return {'take_id':take,'directory':pending,'metadata':metadata}
        except BaseException:
            shutil.rmtree(pending,ignore_errors=True);raise

    def save(self,pending,participant,label,statement=''):
        if label not in LABELS:raise ValueError('Choose truth or lie')
        if not isinstance(statement,str) or len(statement)>4000:raise ValueError('Statement must be at most 4000 characters')
        manifest=self.initialize();split=self.split_for(participant,manifest)
        source=pending['directory'];metadata=dict(pending['metadata'])
        if not source.is_dir():raise RuntimeError('The recording is no longer available')
        from datetime import datetime,timezone
        metadata.update(participant_id=participant,label=label,split=split,statement=statement.strip(),
            saved_utc=datetime.now(timezone.utc).isoformat(),label_source='participant_self_report',split_unit='participant',split_ratios=manifest['split_ratios'])
        destination=self.root/split/label/pending['take_id']
        # Assign identity before publishing a take; a failed write can leave an
        # unused assignment, but cannot place the same identity in two splits.
        manifest['participants'][participant]=split;write_json(self.root/'dataset.json',manifest)
        write_json(source/'metadata.json',metadata)
        source.rename(destination)
        return {'take_id':pending['take_id'],'participant_id':participant,'label':label,'split':split,
                'path':str(destination),'duration_seconds':7.}
