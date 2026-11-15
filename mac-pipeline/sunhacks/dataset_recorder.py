"""Seven-second collection independent of GPU readiness or inference."""
from __future__ import annotations
import copy
from pathlib import Path
import shutil
import threading
import time

import numpy as np
from .dataset_store import DatasetStore,PARTICIPANT
from .temporal import AudioBuffer

ACTIVE=frozenset(('starting','recording','finishing','encoding','awaiting_label','saving','cancelling'))


class DatasetRecorder:
    def __init__(self,pipeline,*,capture_factory=None,store=None):
        self.pipeline=pipeline
        self.store=store or DatasetStore(pipeline.config.get('dataset_root',Path.cwd()/'datasets'/'webcam-seven-second-v1'))
        self._capture_factory=capture_factory
        self._lock=threading.RLock();self._operations=threading.Lock()
        self._cancel=threading.Event();self._capture=None;self._thread=None;self._pending=None;self._closed=False
        self._counts=self.store.counts()
        self._state={'status':'idle','elapsed_seconds':0.,'duration_seconds':7.,'source_frames':0}

    def is_active(self):
        with self._lock:
            if self._state['status'] in ACTIVE or (self._thread and self._thread.is_alive()):return True
            capture=self._capture
        return bool(capture and any(v.get('status') in ('starting','running','stopping') for v in capture.snapshot().values()))

    def snapshot(self):
        with self._lock:
            value=copy.deepcopy(self._state)
            value.update(root=str(self.store.root),counts=copy.deepcopy(self._counts),split_ratios={'train':75,'validation':15,'test':10},
                         split_unit='participant',active=self.is_active(),closed=self._closed)
            capture=self._capture
        value['capture']=capture.snapshot() if capture else {}
        return value

    def start(self,participant):
        if not isinstance(participant,str) or not PARTICIPANT.fullmatch(participant):
            raise ValueError('Enter a participant code (1–64 letters, numbers, underscores or hyphens)')
        with self._operations,self.pipeline.control_lock:
            if self._closed or self.pipeline.closed:raise RuntimeError('Recorder is closed')
            if self.is_active():raise RuntimeError('Label or discard the current recording first')
            showcase=getattr(self.pipeline,'showcase',None)
            if showcase and showcase.is_active():raise RuntimeError('Finish or cancel the showcase before collecting data')
            state=self.pipeline.state()
            if (state['mode']=='demo' or state.get('answer',{}).get('status') in ('capturing','finishing','processing')
                    or any(v.get('status') in ('starting','running','stopping') for v in state.get('capture',{}).values())
                    or any(t.is_alive() for t in self.pipeline.threads)):
                raise RuntimeError('Stop the current test before collecting data')
            lifecycle=state.get('runtime',{}).get('lifecycle',{})
            if (lifecycle.get('active_jobs') or lifecycle.get('active_prepares')
                    or (self.pipeline.prepare_thread and self.pipeline.prepare_thread.is_alive())):
                raise RuntimeError('Wait for current model work to finish')
            if not shutil.which('ffmpeg'):raise RuntimeError('Install ffmpeg before recording dataset clips')
            self.store.initialize()
            if shutil.disk_usage(self.store.root).free<512*1024**2:raise RuntimeError('At least 512 MB free disk space is required')
            factory=self._capture_factory
            if factory is None:
                from .capture import LiveCapture
                factory=LiveCapture
            cfg=self.pipeline.config
            capture=factory(camera_index=cfg.get('camera_device',0),audio_device=cfg.get('audio_device'),
                camera_fps=cfg.get('capture_fps',30),max_frame_edge=cfg.get('capture_max_frame_edge',640))
            with self._lock:
                self._cancel=threading.Event();self._capture=capture;self._pending=None
                self._state={'status':'starting','elapsed_seconds':0.,'duration_seconds':7.,'source_frames':0,
                             'participant_id':participant,'message':'Opening camera and microphone. Wait for the recording timer.'}
                self._thread=threading.Thread(target=self._record,args=(capture,participant,self._cancel),name='dataset-recording',daemon=True)
                self._thread.start()
        return self.snapshot()

    def _record(self,capture,participant,cancel):
        frames=[];audio=AudioBuffer(retention_seconds=18);start=None;pending=None;error=None
        started=time.monotonic();devices={}
        try:
            if cancel.is_set():return
            capture.start()
            while not cancel.is_set():
                now=time.monotonic();video,chunks=capture.drain()
                for chunk in chunks:audio.add(chunk)
                if audio.chunks:
                    audio_start=audio.chunks[0].start
                    for frame in video:
                        if frame.timestamp>=audio_start and (not frames or frame.timestamp>frames[-1].timestamp):
                            if start is None:start=frame.timestamp
                            if frame.timestamp<start+7:frames.append(frame)
                if len(frames)>600:raise RuntimeError('Unexpected frame rate exceeded the recording buffer')
                devices=capture.snapshot()
                for modality in ('camera','audio'):
                    if devices.get(modality,{}).get('status') in ('missing','error'):
                        raise RuntimeError(modality+': '+devices[modality].get('error','Device unavailable'))
                if start is None:
                    if now-started>8:raise RuntimeError('Camera and microphone did not become ready within eight seconds')
                else:
                    elapsed=min(7.,max(0.,now-start))
                    with self._lock:self._state.update(status='recording' if elapsed<7 else 'finishing',elapsed_seconds=elapsed,
                        source_frames=len(frames),message='Speak now — recording seven seconds.' if elapsed<7 else 'Stopping capture…')
                    if now>=start+7:
                        waveform,quality=audio.slice(start,start+7)
                        if waveform is not None:break
                        if now>=start+7.75:raise RuntimeError('Microphone has missing samples; discard and try again')
                cancel.wait(.02)
            if cancel.is_set():return
            capture.stop()
            if any(v.get('status') in ('running','starting','stopping') for v in capture.snapshot().values()):
                raise RuntimeError('Capture did not stop; restart the application before recording again')
            if len(frames)<128 or frames[-1].timestamp-start<6.7:raise RuntimeError('Not enough video frames for a complete seven-second recording')
            if max(np.diff([f.timestamp for f in frames]))>.25:raise RuntimeError('Camera gap exceeded 250 ms; recording was not saved')
            if any(devices.get(k,{}).get(field,0) for k,field in (('camera','queue_dropped'),('audio','queue_dropped'),
                ('audio','overflow_callbacks'),('audio','adc_timeline_gap_callbacks'),('audio','resampling_discontinuities'))):
                raise RuntimeError('Capture dropped data; recording was not saved')
            with self._lock:self._state.update(status='encoding',message='Capture stopped. Preparing the recording for review…')
            pending=self.store.encode(tuple(frames),waveform,start,devices,cancelled=cancel)
            with self._lock:
                if not cancel.is_set():
                    self._pending=pending
                    self._state.update(status='awaiting_label',take_id=pending['take_id'],elapsed_seconds=7.,
                        message='Was your statement truthful or a lie? Choose its actual label, or discard it.')
        except Exception as exc:error=str(exc)[:400]
        finally:
            capture.stop();frames.clear();audio.chunks.clear()
            with self._lock:
                if cancel.is_set():
                    if pending:shutil.rmtree(pending['directory'],ignore_errors=True)
                    self._pending=None;self._state.update(status='idle',message='Recording discarded.',take_id=None)
                elif error:self._state.update(status='error',message=error)

    def save(self,take_id,label,statement=''):
        with self._operations,self._lock:
            if self._state['status']!='awaiting_label' or not self._pending:raise RuntimeError('No recording is ready to label')
            if not isinstance(take_id,str) or take_id!=self._pending['take_id']:raise ValueError('Recording changed; refresh before labeling')
            result=self.store.save(self._pending,self._state['participant_id'],label,statement)
            self._pending=None;self._counts=self.store.counts()
            self._state.update(status='saved',saved=result,message=f"Saved {label} to {result['split']}/{label}.")
        return self.snapshot()

    def cancel(self):
        with self._operations,self._lock:
            self._cancel.set()
            if self._pending:shutil.rmtree(self._pending['directory'],ignore_errors=True)
            self._pending=None
            self._state.update(status='cancelling' if self._thread and self._thread.is_alive() else 'idle',message='Discarding recording…',take_id=None)
        return self.snapshot()

    def clear_capture_error(self,message):
        with self._operations,self._lock:
            if self.is_active():raise RuntimeError('Stop capture before clearing its error')
            if self._state['status']=='error':
                self._capture=None
                self._state={'status':'idle','elapsed_seconds':0.,'duration_seconds':7.,'source_frames':0,'message':message}

    def preview(self,take_id):
        with self._lock:
            if not self._pending or self._state['status']!='awaiting_label' or take_id!=self._pending['take_id']:
                raise ValueError('Recording preview is no longer available')
            return (self._pending['directory']/'video.mp4').read_bytes()

    def close(self):
        with self._lock:self._closed=True
        self.cancel()
        if self._thread:self._thread.join(timeout=3)
