"""Video-only GPU RPC; resident local audio/text, real digest and exported stack.

The video classifier is an independent output and is never inserted into the
learned two-branch Dasheng/digest stack. No raw audio or text argument is passed
to the video client. One source job per component bounds outstanding work; late
features are discarded and model release waits for all forwards to finish.
"""
from __future__ import annotations
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import threading
import time
import numpy as np

from .asr import prepare_asr, transcribe_local, clear_asr_cache
from .frozen_features import FrozenFeatureModels
from .local_digest import build_digest_row, prepare_digest_faces, digest_contract
from .local_faces import LocalFacePreprocessor
from .local_stack_process import ProcessDigestStack

BRANCHES=("vision","dasheng","text","emotion","tabular","stack","fusion")
LOCALITY={"vision":"friend_gpu_video_only","asr":"mac","audio":"mac","text":"mac",
          "emotion":"mac","digest":"mac","ensemble":"mac",
          "audio_transmitted":False,"transcript_transmitted":False}


class SplitRuntime:
    def __init__(self,config,video_client=None,on_progress=None,*,dasheng=None,
                 feature_models=None,face_preprocessor=None,digest_stack=None,
                 prepare_asr_fn=None,transcribe_fn=None,cleanup_asr_fn=None):
        self.config=copy.deepcopy(config)
        self.classifier_profile=self.config.get("classifier_profile","courtroom")
        if self.classifier_profile not in ("courtroom","webcam_pilot"):raise ValueError("Unknown classifier profile")
        self.video_client=video_client
        self.on_progress=on_progress
        self.dasheng=dasheng
        self.feature_models=feature_models
        self.faces=face_preprocessor
        self.digest_stack=digest_stack
        self.asr_config=self.config.get("models",{}).get("asr",self.config.get("asr",{})).copy()
        self.asr_config["keep_loaded"]=True
        self._asr_prepare=prepare_asr_fn or prepare_asr
        self._transcribe=transcribe_fn or transcribe_local
        self._clear_asr=cleanup_asr_fn or clear_asr_cache
        self.options=self.config.get("split_runtime",self.config.get("fast_runtime",{}))
        self.deadline_seconds=float(self.options.get("deadline_seconds",30))
        if not np.isfinite(self.deadline_seconds) or not .01<=self.deadline_seconds<=120:
            raise ValueError("Split runtime deadline must be within0.01..120seconds")
        self._lock=threading.RLock()
        self._mps_lock=threading.RLock()
        self._inflight={}
        self._active_prepares=0
        self._closed=False
        self._cleanup_started=False
        self._cleanup_done=False
        self._cleanup_errors=[]
        self._executor=ThreadPoolExecutor(max_workers=5,thread_name_prefix="sunhacks-split")
        self.preparation={"status":"not_prepared","local_weights_loaded":False,"components":{}}

    def _progress(self,stage,status,**details):
        if self.on_progress:
            try:self.on_progress({"stage":stage,"status":status,**details})
            except Exception:pass

    def _stopped(self,stop_event=None,deadline=None):
        return self._closed or (stop_event is not None and stop_event.is_set()) or (deadline is not None and time.monotonic()>=deadline)

    @staticmethod
    def _missing(branch,reason,window_id=None,status="unavailable"):
        return {"status":status,"model_id":branch,"source_window_id":window_id,
                "raw_score":None,"embedding_shape":None,"elapsed_ms":0.0,"reason":reason,
                "calibrated":False,"locality":"friend_gpu_video_only" if branch=="vision" else "mac"}

    @staticmethod
    def _check_source(value, window_id, component):
        if not isinstance(value,dict):
            raise ValueError(component+" returned an invalid component result")
        supplied=value.get("source_window_id")
        if supplied is not None and (isinstance(supplied,(bool,np.bool_)) or not isinstance(supplied,(int,np.integer)) or supplied!=window_id):
            raise ValueError(component+" returned a different source answer; output discarded")

    def _make_components(self):
        paths=self.config.get("local_ensemble",{})
        if self.dasheng is None:
            from .local_dasheng import TunedDashengEnsemble
            cfg=self.config.get("models",{}).get("dasheng",{})
            kwargs={"manifest_path":paths["manifest_path"],"fold_dir":paths["fold_dir"],
                    "base_path":cfg["path"],"source_path":cfg.get("source_path",cfg["path"])}
            if paths.get("expected_manifest_sha256"):kwargs["expected_manifest_sha256"]=paths["expected_manifest_sha256"]
            if self.classifier_profile=="webcam_pilot":
                from .pilot_audio import PilotDasheng
                pilot=self.config.get("pilot_audio",{})
                self.dasheng=PilotDasheng(pilot["manifest_path"],pilot["fold_dir"],base_path=cfg["path"],source_path=cfg.get("source_path",cfg["path"]))
            else:self.dasheng=TunedDashengEnsemble(**kwargs)
        if self.feature_models is None:self.feature_models=FrozenFeatureModels(self.config)
        if self.faces is None:
            self.faces=LocalFacePreprocessor(paths["face_model_path"],expected_sha256=paths.get("expected_face_sha256"))
        if self.digest_stack is None:
            kwargs={}
            for key in ("feature_version","schema_path","expected_schema_sha256"):
                if paths.get(key) is not None:kwargs[key]=paths[key]
            if paths.get("expected_manifest_sha256"):kwargs["expected_manifest_sha256"]=paths["expected_manifest_sha256"]
            if paths.get("expected_xgb_sha256"):kwargs["expected_model_sha256"]=paths["expected_xgb_sha256"]
            self.digest_stack=ProcessDigestStack(paths["manifest_path"],paths["xgb_path"],**kwargs)

    def prepare(self):
        """Prepare local weights/static intents; no sensor use or video request.

        Static Qwen intent preparation is an actual local forward, distinguished
        from merely loading weights. A remote health probe remains explicit in
        the caller; this method only reads the client's existing snapshot.
        """
        started=time.perf_counter()
        with self._lock:
            if self._closed:return {"status":"stopped","components":{},"local_weights_loaded":False}
            if self._active_prepares or self._inflight:
                return {"status":"busy","components":{},"local_weights_loaded":False}
            self._active_prepares+=1
        components={}
        try:
            self._progress("local_models","loading")
            try:self._make_components()
            except Exception as exc:
                return self._prepared_result(components,started,error=exc)
            if self.asr_config.get("path"):
                with self._mps_lock:
                    components["asr"]=self._asr_prepare(self.asr_config)
            else:components["asr"]={"status":"unconfigured","reason":"A same-answer manual transcript is required when local ASR is off."}
            if self._stopped():return self._prepared_result(components,started)
            for name,component in (("dasheng",self.dasheng),("faces",self.faces),("digest",self.digest_stack)):
                self._progress(name,"loading")
                try:
                    components[name]=component.prepare()
                except Exception as exc:components[name]={"status":"unavailable","reason":type(exc).__name__+": "+str(exc)[:300]}
            if not self._stopped():
                with self._mps_lock:
                    self._progress("frozen_features","preparing_static_intents")
                    try:
                        result=self.feature_models.prepare()
                        components.update(result.get("components",{}))
                    except Exception as exc:
                        for name in ("text","emotion"):components[name]={"status":"unavailable","reason":type(exc).__name__+": "+str(exc)[:300]}
            return self._prepared_result(components,started)
        finally:
            with self._lock:self._active_prepares-=1
            self._cleanup_if_ready()

    def _prepared_result(self,components,started,error=None):
        needed=("dasheng","faces","digest","text","emotion")
        local=all(components.get(k,{}).get("status")=="ready" for k in needed)
        if self.asr_config.get("path"):local=local and components.get("asr",{}).get("status")=="ready"
        remote={"classifier_ready":False,"reason":"Video client is not configured"}
        if self.video_client is not None:
            try:remote=self.video_client.snapshot()
            except Exception:remote={"classifier_ready":False,"reason":"Video client snapshot unavailable"}
        value={"status":"stopped" if self._closed else "ready" if local else "partial",
               "components":components,"local_weights_loaded":local,"locality":dict(LOCALITY),
               "remote":remote,"remote_verified":False,
               "kernels_warmed":False,"static_qwen_intent_forward":components.get("text",{}).get("kernels_warmed",False),
               "elapsed_ms":round((time.perf_counter()-started)*1000,3)}
        if error:value["reason"]=type(error).__name__+": "+str(error)[:300]
        with self._lock:self.preparation=copy.deepcopy(value)
        return value

    def _done(self,key,future):
        with self._lock:
            if self._inflight.get(key) is future:self._inflight.pop(key,None)
        self._cleanup_if_ready()

    def _submit(self,key,function):
        with self._lock:
            if self._closed:return None,"stopped"
            if self._active_prepares:return None,"preparing"
            if key in self._inflight and not self._inflight[key].done():return None,"busy"
            future=self._executor.submit(function)
            self._inflight[key]=future
            future.add_done_callback(lambda done:self._done(key,done))
            return future,"submitted"

    def _semantic(self,audio,transcript,window_id,stop_event,deadline,on_transcript=None):
        acquired=self._mps_lock.acquire(timeout=max(0,deadline-time.monotonic()))
        if not acquired:raise TimeoutError("Semantic path could not acquire MPS before deadline")
        try:
            if self._stopped(stop_event,deadline):raise RuntimeError("Source answer stopped")
            value=copy.deepcopy(transcript)
            used_asr=audio is not None and bool(self.asr_config.get("path"))
            if used_asr:
                self._progress("local_transcription","running",window_id=window_id)
                value=self._transcribe(audio,self.asr_config)
                self._check_source(value,window_id,"transcript")
                value.update(source="local_asr",scope="video_window",source_window_id=window_id)
            else:
                self._check_source(value,window_id,"transcript")
            if value.get("status") not in ("available","base") or not value.get("text","").strip():
                return {"transcript":value,"text":self._missing("text","Complete answer transcript is unavailable",window_id)}
            if not self.asr_config.get("path") and value.get("scope") not in ("video_window","answer","same_answer"):
                return {"transcript":value,"text":self._missing("text","Manual session-context text is not a verified transcript of this answer",window_id)}
            if self._stopped(stop_event,deadline):raise RuntimeError("Source answer stopped before Qwen")
            hook_error=None
            hook_dispatched=False
            if on_transcript is not None and used_asr:
                # This is a dispatch seam: the callback must submit its work and
                # return immediately. Its returned future/result is never waited
                # on here. A copy prevents a caller from altering decoded words
                # or source identity used by Qwen and the local digest.
                try:
                    on_transcript(copy.deepcopy(value))
                    hook_dispatched=True
                except Exception as exc:
                    hook_error={"source_window_id":window_id,"error_type":type(exc).__name__,
                                "reason":"Transcript guard dispatch failed."}
            if self._stopped(stop_event,deadline):raise RuntimeError("Source answer stopped before Qwen")
            self._progress("text","running",window_id=window_id)
            result={"transcript":value,"text":self.feature_models.text(value["text"])}
            if on_transcript is not None and used_asr:
                result["claim_hook_dispatched"]=hook_dispatched
            if hook_error is not None:result["claim_hook_error"]=hook_error
            return result
        finally:self._mps_lock.release()

    def infer(self,clip,audio,transcript,window_id,*,timestamps,duration_s=None,stop_event=None,on_transcript=None,video_frames=None):
        """Run one answer; optionally dispatch a guard after valid same-answer ASR.

        ``on_transcript(copy_of_validated_transcript)`` is ASR-only, called once
        before Qwen. It must return immediately; any returned future is ignored.
        Callback errors are diagnostics and do not disable normal local models.
        """
        started=time.monotonic();deadline=started+self.deadline_seconds
        branches={k:self._missing(k,"This source answer has not supplied compatible inputs",window_id) for k in BRANCHES}
        branches["fusion"]["reason"]="The exported two-branch stack does not use the old ReDA fusion or video classifier score."
        source_transcript=copy.deepcopy(transcript)
        output={"branches":branches,"features":{},"transcript":copy.deepcopy(source_transcript),
                "locality":dict(LOCALITY),"timings":{"jobs":{},"deadline_ms":self.deadline_seconds*1000}}
        video_enabled=self.config.get("remote_vision",{}).get("enabled",True)
        output["remote_video_enabled"]=video_enabled
        if not video_enabled:
            branches["vision"]={**self._missing("vision","GPU video is disabled for this answer",window_id),"status":"disabled"}
            output["locality"]["vision"]="disabled"
        if self._stopped(stop_event):return self._finish_output(output,started,window_id,stop_event)
        if not self.preparation.get("local_weights_loaded"):
            output["reason"]="Local split-runtime components are not prepared"
            return self._finish_output(output,started,window_id,stop_event)
        ts=np.asarray(timestamps,dtype=np.float64)
        frames=np.asarray(clip)
        if (frames.ndim!=4 or frames.dtype!=np.uint8 or frames.shape[0]!=128 or frames.shape[-1]!=3 or
                ts.shape!=(128,) or not np.isfinite(ts).all() or np.any(np.diff(ts)<0) or ts[-1]<=ts[0]):
            raise ValueError("Split runtime requires one already-selected128-frame RGB answer with its unchanged nondecreasing timestamps")
        waveform=None if audio is None else np.asarray(audio,dtype=np.float32)
        if waveform is not None and (waveform.ndim!=1 or not len(waveform) or not np.isfinite(waveform).all()):
            raise ValueError("Audio must be complete finite mono16kHz float samples")
        duration=float(duration_s if duration_s is not None else len(waveform)/16000 if waveform is not None else ts[-1]-ts[0])
        if not np.isfinite(duration) or duration<=0:raise ValueError("Actual source answer duration must be positive")
        def video():
            if self.video_client is None:raise RuntimeError("Video-only client is unavailable")
            if video_frames is None:
                raw_frames,raw_ts=frames,ts
            else:
                raw_frames=np.stack([f.rgb for f in video_frames])
                raw_ts=np.asarray([f.timestamp for f in video_frames],dtype=np.float64)
                from .local_digest import select_timestamp_frames
                idx,selected_ts=select_timestamp_frames(raw_ts)
                if not np.array_equal(selected_ts,ts) or not np.array_equal(raw_frames[idx],frames):
                    raise ValueError("Remote raw capture and local cache must select the exact same source frames")
            return self.video_client.predict(raw_frames,window_id=window_id,timestamps=raw_ts,face_crop=True)
        def face():
            return prepare_digest_faces(frames,ts,self.faces,already_selected=True)
        def dasheng():
            if waveform is None:raise ValueError("Audio modality is missing")
            return self.dasheng.infer(waveform,sample_rate=16000,source_window_id=window_id,stop_event=stop_event)
        def emotion():
            if waveform is None:raise ValueError("Audio modality is missing")
            return self.feature_models.emotion(waveform)
        specs={"vision":video,"dasheng":dasheng,"semantic":lambda:self._semantic(waveform,source_transcript,window_id,stop_event,deadline,on_transcript),"emotion":emotion,"face":face}
        if not video_enabled:specs.pop("vision")
        pending={};results={}
        for key,fn in specs.items():
            def guarded(fn=fn):
                if self._stopped(stop_event,deadline):raise RuntimeError("Source answer stopped before component started")
                value=fn()
                if self._stopped(stop_event,deadline):raise RuntimeError("Late source-answer component output discarded")
                return value
            if self._stopped(stop_event,deadline):future,status=None,"stopped"
            else:future,status=self._submit(key,guarded)
            output["timings"]["jobs"][key]={"status":status}
            if future:pending[future]=key
            elif key=="semantic":
                output["transcript"]={"status":status,"text":"","source_window_id":window_id}
                branches["text"]=self._missing("text","Component is "+status+"; no previous source transcript is reused",window_id)
            elif key in branches:branches[key]=self._missing(key,"Component is "+status+"; no previous source features are reused",window_id)
        while pending and not self._stopped(stop_event,deadline):
            done,_=wait(pending,timeout=min(.05,max(0,deadline-time.monotonic())),return_when=FIRST_COMPLETED)
            for future in done:
                key=pending.pop(future)
                try:
                    value=future.result()
                    if key=="semantic":
                        self._check_source(value["transcript"],window_id,"transcript")
                        self._check_source(value["text"],window_id,"text")
                    elif key in branches:self._check_source(value,window_id,key)
                    results[key]=value
                    output["timings"]["jobs"][key]["status"]="complete"
                    if key=="semantic":
                        output["transcript"],branches["text"]=value["transcript"],value["text"]
                        if "claim_hook_dispatched" in value:output["claim_hook_dispatched"]=value["claim_hook_dispatched"]
                        if "claim_hook_error" in value:output["claim_hook_error"]=copy.deepcopy(value["claim_hook_error"])
                    elif key in branches:branches[key]=value
                except Exception as exc:
                    output["timings"]["jobs"][key].update(status="error",reason=type(exc).__name__+": "+str(exc)[:240])
                    branch="text" if key=="semantic" else key
                    if key=="semantic":output["transcript"]={"status":"error","text":"","source_window_id":window_id}
                    if branch in branches:branches[branch]=self._missing(branch,type(exc).__name__+": "+str(exc)[:300],window_id,"error")
        for future,key in pending.items():
            future.cancel();output["timings"]["jobs"][key]["status"]="stopped" if self._stopped(stop_event) else "timeout"
            branch="text" if key=="semantic" else key
            if key=="semantic":output["transcript"]={"status":"unavailable","text":"","source_window_id":window_id}
            if branch in branches:branches[branch]=self._missing(branch,"Source answer stopped or shared deadline expired; late result discarded",window_id)
        if not self._stopped(stop_event,deadline):
            try:
                if any(k not in results for k in ("dasheng","semantic","emotion","face")):
                    raise ValueError("A required local modality did not complete for this answer")
                text=branches["text"]
                if text.get("status") not in ("base","trained"):raise ValueError("Complete same-answer Qwen features are missing")
                crops,meta=results["face"]
                feature_version=self.config.get("local_ensemble",{}).get("feature_version","v2-no-within-z")
                row=build_digest_row(transcript=output["transcript"]["text"],audio=waveform,duration_s=duration,
                    face_crops=crops,face_metadata=meta,emotion_probabilities=branches["emotion"].get("_class_probabilities"),
                    qwen_embedding=text.get("_full_embedding"),intent_embeddings=text.get("_intent_embeddings"),
                    feature_version=feature_version)
                scored=self.digest_stack.infer(row,branches["dasheng"].get("raw_score"),source_window_id=window_id)
                for name in ("tabular","stack"):self._check_source(scored[name],window_id,name)
                branches.update(scored)
                output["features"]={"digest":{"status":"available","feature_count":digest_contract(feature_version)["feature_count"],"feature_version":feature_version,"values":row.public_features(),"missing_features":list(row.missing_features)},
                    "face":{"status":"available","face_detect_rate":meta["face_detect_rate"],"centre_crop_fallback":meta.get("centre_crop_fallback")}}
            except Exception as exc:
                for branch in ("tabular","stack"):branches[branch]=self._missing(branch,type(exc).__name__+": "+str(exc)[:300],window_id)
        if self.classifier_profile=="webcam_pilot":
            # The pilot has its own calibrated probability and cutoff. Never
            # reuse the courtroom stack/cutoff if the pilot forward failed.
            decision=branches["dasheng"].get("_pilot_decision")
            branches["stack"]=copy.deepcopy(decision) if isinstance(decision,dict) else self._missing("stack","Selected pilot audio model did not complete; no courtroom fallback",window_id)
            if "face" in results and "face" not in output["features"]:
                _,meta=results["face"]
                output["features"]["face"]={"status":"available","face_detect_rate":meta["face_detect_rate"],"centre_crop_fallback":meta.get("centre_crop_fallback")}
        return self._finish_output(output,started,window_id,stop_event)

    def _finish_output(self,output,started,window_id,stop_event):
        stopped=self._stopped(stop_event)
        for key,value in output["branches"].items():
            if stopped:
                output["branches"][key]=self._missing(key,"Source answer stopped; outputs discarded",window_id,"cancelled")
                continue
            try:self._check_source(value,window_id,key)
            except ValueError as exc:
                output["branches"][key]=self._missing(key,str(exc),window_id,"error")
                continue
            value["source_window_id"]=window_id
        if stopped:output["features"]={};output["transcript"]={"status":"cancelled","text":"","source_window_id":window_id}
        output["timings"]["service_ms"]=round((time.monotonic()-started)*1000,3)
        return output

    def snapshot(self):
        with self._lock:return {"closed":self._closed,"cleanup_done":self._cleanup_done,
            "cleanup_in_progress":self._cleanup_started and not self._cleanup_done,
            "active_jobs":sorted(self._inflight),"active_prepares":self._active_prepares,
            "preparation":copy.deepcopy(self.preparation),"locality":dict(LOCALITY),
            "cleanup_errors":list(self._cleanup_errors)}

    def _cleanup_if_ready(self):
        with self._lock:
            if not self._closed or self._cleanup_started or self._inflight or self._active_prepares:return
            self._cleanup_started=True
        errors=[]
        try:
            for component in (self.dasheng,self.feature_models,self.faces,self.digest_stack):
                if component is not None:
                    try:component.close()
                    except Exception as exc:errors.append(type(exc).__name__+": "+str(exc)[:160])
            try:self._clear_asr()
            except Exception as exc:errors.append(type(exc).__name__+": "+str(exc)[:160])
        finally:
            with self._lock:
                self._cleanup_errors=errors
                self._cleanup_done=True

    def close(self):
        with self._lock:
            self._closed=True
            for future in tuple(self._inflight.values()):future.cancel()
        if self.video_client is not None:self.video_client.close()
        self._executor.shutdown(wait=False,cancel_futures=True)
        self._cleanup_if_ready()
        state=self.snapshot()
        return {"status":"cleanup_error" if state["cleanup_errors"] else "closed" if state["cleanup_done"] else "closing",**state}
