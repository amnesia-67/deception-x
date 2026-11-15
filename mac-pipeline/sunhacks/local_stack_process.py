"""Local digest inference isolated from Torch/MediaPipe native runtimes.

Parent imports no XGBoost. Its persistent owned child receives only the real
versioned row over bounded private stdio, with undefined fields as JSON null.
The child has an allowlisted environment and no API keys; its code performs no network operations.
Model/schema hashes are checked independently on both sides. Platt/stacking
stays in this parent process. A failed or timed-out child is never reused.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading
import time
import uuid
import numpy as np

from .local_digest import combine_ensemble_scores
from .local_stack import LocalDigestStack,MANIFEST_SHA256,MODEL_SHA256

PROTOCOL="sunhacks-xgb-worker-v1"
MAX_LINE_BYTES=65536


class DigestWorkerError(RuntimeError):
    pass


def _pairs(items):
    result={}
    for key,value in items:
        if key in result:raise DigestWorkerError("Duplicate worker response field")
        result[key]=value
    return result


def worker_environment():
    """Allow only ordinary runtime settings; never inherit provider credentials."""
    out={key:os.environ[key] for key in ("PATH","LANG","LC_ALL","LC_CTYPE","TMPDIR","SYSTEMROOT","WINDIR") if key in os.environ}
    out.update(PYTHONUNBUFFERED="1",PYTHONNOUSERSITE="1",OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    return out


class ProcessDigestStack(LocalDigestStack):
    def __init__(self,manifest_path,model_path,*,expected_manifest_sha256=MANIFEST_SHA256,
                 expected_model_sha256=MODEL_SHA256,schema_path=None,
                 expected_schema_sha256=None,feature_version="v2-no-within-z",
                 timeout_seconds=5.0,_worker_path=None):
        super().__init__(manifest_path,model_path,expected_manifest_sha256=expected_manifest_sha256,
                         expected_model_sha256=expected_model_sha256,feature_version=feature_version,
                         schema_path=schema_path,expected_schema_sha256=expected_schema_sha256)
        if isinstance(timeout_seconds,bool) or not isinstance(timeout_seconds,(int,float)) or not math.isfinite(timeout_seconds) or not .01<=timeout_seconds<=30:
            raise ValueError("Worker timeout must be within0.01..30seconds")
        self.timeout=float(timeout_seconds)
        self.worker_path=Path(_worker_path) if _worker_path is not None else Path(__file__).with_name('_xgb_worker.py')
        self._names_sha=hashlib.sha256(json.dumps(self.schema['feature_names'],separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
        self._process=None
        self._rpc_lock=threading.Lock()
        self._lifecycle_lock=threading.RLock()
        self._closed=False
        self._worker_info=None

    def _response_has_schema(self,response,*,require_count=False):
        # Old fake/legacy workers omitted version, and prediction replies also
        # omitted count. New-schema replies must identify both explicitly.
        version=response.get("feature_version")
        count=response.get("feature_count")
        if self.feature_version=="v2-no-within-z":
            if version is None:version="v2-no-within-z"
            if count is None and not require_count:count=89
        return version==self.feature_version and type(count) is int and count==self.feature_count

    def _remaining(self,deadline):
        value=deadline-time.monotonic()
        if value<=0:raise DigestWorkerError("Local XGBoost worker deadline exceeded")
        return value

    @contextmanager
    def _rpc_guard(self,deadline):
        if not self._rpc_lock.acquire(timeout=self._remaining(deadline)):
            raise DigestWorkerError("Local XGBoost worker is busy past the request deadline")
        try:
            if self._closed:raise DigestWorkerError("Local XGBoost worker is closed")
            yield
        finally:
            self._rpc_lock.release()

    def _exchange(self,process,payload,*,timeout_seconds=None):
        deadline=time.monotonic()+(self.timeout if timeout_seconds is None else min(self.timeout,timeout_seconds))
        request={**payload,"protocol":PROTOCOL,"request_id":uuid.uuid4().hex}
        data=json.dumps(request,allow_nan=False,separators=(',',':')).encode()+b'\n'
        if len(data)>MAX_LINE_BYTES:raise DigestWorkerError("Local XGBoost request exceeds byte limit")
        try:
            if process.poll() is not None:raise DigestWorkerError("Local XGBoost worker exited")
            with selectors.DefaultSelector() as selector:
                fd=process.stdin.fileno();selector.register(fd,selectors.EVENT_WRITE)
                position=0
                while position<len(data):
                    if self._closed:raise DigestWorkerError("Local XGBoost worker is closed")
                    if not selector.select(self._remaining(deadline)):raise DigestWorkerError("Local XGBoost worker write timed out")
                    try:written=os.write(fd,data[position:])
                    except (BlockingIOError,InterruptedError):continue
                    if written<=0:raise DigestWorkerError("Local XGBoost worker pipe closed")
                    position+=written
                selector.unregister(fd)
                fd=process.stdout.fileno();selector.register(fd,selectors.EVENT_READ)
                response=bytearray()
                while b'\n' not in response:
                    if self._closed:raise DigestWorkerError("Local XGBoost worker is closed")
                    if not selector.select(self._remaining(deadline)):raise DigestWorkerError("Local XGBoost worker read timed out")
                    try:chunk=os.read(fd,min(16384,MAX_LINE_BYTES+1-len(response)))
                    except (BlockingIOError,InterruptedError):continue
                    if not chunk:raise DigestWorkerError("Local XGBoost worker exited without a response")
                    response.extend(chunk)
                    if len(response)>MAX_LINE_BYTES:raise DigestWorkerError("Local XGBoost response exceeds byte limit")
                line,extra=bytes(response).split(b'\n',1)
                if extra:raise DigestWorkerError("Local XGBoost worker returned multiple responses")
            result=json.loads(line.decode('utf-8'),object_pairs_hook=_pairs,
                parse_constant=lambda value:(_ for _ in ()).throw(DigestWorkerError("Nonfinite worker JSON")))
            if (not isinstance(result,dict) or result.get('protocol')!=PROTOCOL or
                    result.get('request_id')!=request['request_id']):
                raise DigestWorkerError("Local XGBoost worker response identity mismatch")
            if result.get('status')=='error':raise DigestWorkerError("Local XGBoost worker rejected the request: "+str(result.get('code','error'))[:80])
            return result
        except DigestWorkerError:raise
        except (OSError,ValueError,TypeError,UnicodeError,RecursionError) as exc:
            raise DigestWorkerError("Local XGBoost worker IPC failed: "+type(exc).__name__) from None

    @staticmethod
    def _terminate(process):
        if process is None:return
        if process.poll() is None:
            try:process.terminate()
            except (ProcessLookupError,OSError):pass
        try:process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:process.kill()
            except (ProcessLookupError,OSError):pass
            try:process.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
        for pipe in (process.stdin,process.stdout):
            if pipe is not None:
                try:pipe.close()
                except OSError:pass

    def _invalidate(self,process):
        with self._lifecycle_lock:
            if self._process is process:
                self._process=None;self.manifest=None;self.model=None;self._worker_info=None
        self._terminate(process)

    def prepare(self):
        started=time.perf_counter();deadline=time.monotonic()+self.timeout
        with self._rpc_guard(deadline):
            with self._lifecycle_lock:
                if self._closed:raise DigestWorkerError("Local XGBoost worker is closed")
                process=self._process
            if process is not None and process.poll() is None and self.manifest is not None:
                return self._prepared(started,cache_hit=True)
            if process is not None:self._invalidate(process)
            manifest=self._validated_manifest()
            self._read(self.model_path,self.model_sha,4*1024*1024)
            schema=json.loads(self._read(self.schema_path,self.schema_sha,65536))
            if schema.get('feature_names')!=self.schema['feature_names'] or schema.get('feature_version')!=self.feature_version:
                raise DigestWorkerError("Local XGBoost schema does not match its reviewed version/fields")
            with self._lifecycle_lock:
                if self._closed:raise DigestWorkerError("Local XGBoost worker is closed")
                process=subprocess.Popen([sys.executable,'-u',str(self.worker_path)],stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0,env=worker_environment(),close_fds=True)
                self._process=process
                os.set_blocking(process.stdin.fileno(),False);os.set_blocking(process.stdout.fileno(),False)
            try:
                response=self._exchange(process,{"action":"prepare","model_path":str(self.model_path.resolve()),
                    "model_sha256":self.model_sha,"schema_path":str(self.schema_path.resolve()),
                    "schema_sha256":self.schema_sha,"feature_names_sha256":self._names_sha,
                    "feature_version":self.feature_version,"feature_count":self.feature_count},timeout_seconds=self._remaining(deadline))
                if (response.get('status')!='ready' or type(response.get('pid')) is not int or response['pid']!=process.pid or
                        not self._response_has_schema(response,require_count=True) or response.get('model_sha256')!=self.model_sha or
                        response.get('schema_sha256')!=self.schema_sha or response.get('feature_names_sha256')!=self._names_sha or
                        response.get('torch_imported') is not False or response.get('backend')!='xgboost_cpu_process'):
                    raise DigestWorkerError("Local XGBoost worker handshake contract mismatch")
                with self._lifecycle_lock:
                    if self._closed:raise DigestWorkerError("Local XGBoost worker closed during preparation")
                    self.manifest=manifest;self.model='isolated_process';self._worker_info=response
                return self._prepared(started,cache_hit=False)
            except Exception:
                self._invalidate(process)
                raise

    def _prepared(self,started,cache_hit):
        with self._lifecycle_lock:
            process=self._process
            if self._closed or process is None:raise DigestWorkerError("Local XGBoost worker closed during readiness reporting")
            pid=process.pid
        return {"status":"ready","weights_loaded":True,"kernels_warmed":False,"feature_count":self.feature_count,"feature_version":self.feature_version,
                "locality":"mac","backend":"xgboost_cpu_process","worker_pid":pid,
                "parent_imports_xgboost":False,"worker_imports_torch":False,"cache_hit":cache_hit,
                "elapsed_ms":round((time.perf_counter()-started)*1000,3)}

    def infer(self,row,dasheng_score,*,source_window_id=None,timeout_seconds=None):
        started=time.perf_counter()
        if (tuple(row.feature_names)!=tuple(self.schema['feature_names']) or row.values.shape!=(1,self.feature_count)
                or getattr(row,"feature_version","v2-no-within-z")!=self.feature_version):
            raise ValueError("Digest schema/version/order/shape changed")
        if np.isinf(row.values).any():raise ValueError("Infinite digest feature")
        values=[None if np.isnan(value) else float(value) for value in row.values[0]]
        if timeout_seconds is not None and (isinstance(timeout_seconds,bool) or not isinstance(timeout_seconds,(int,float)) or not math.isfinite(timeout_seconds) or timeout_seconds<=0):
            raise ValueError("Prediction timeout must be positive")
        deadline=time.monotonic()+(self.timeout if timeout_seconds is None else min(self.timeout,timeout_seconds))
        with self._rpc_guard(deadline):
            with self._lifecycle_lock:process=self._process;manifest=self.manifest
            if self._closed or process is None or manifest is None:raise DigestWorkerError("Local XGBoost worker is not prepared")
            try:
                response=self._exchange(process,{"action":"predict","values":values,
                    "source_window_id":source_window_id,"feature_names_sha256":self._names_sha,
                    "feature_version":self.feature_version,"feature_count":self.feature_count},timeout_seconds=self._remaining(deadline))
                value=response.get('raw_score')
                if (response.get('status')!='success' or not self._response_has_schema(response) or response.get('source_window_id')!=source_window_id or type(response.get('source_window_id')) is not type(source_window_id) or
                        isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1):
                    raise DigestWorkerError("Invalid local XGBoost score/source identity")
                score=combine_ensemble_scores(dasheng_score,value,manifest)
            except Exception:
                self._invalidate(process)
                raise
        common={"status":"trained_classifier","classifier_only":True,"calibrated":False,
                "source_window_id":source_window_id,"locality":"mac","precision_parity_verified":False,
                "feature_version":self.feature_version}
        return {"tabular":{**common,"model_id":f"teammate/xgb_digest_{self.feature_count}","raw_score":float(value),
                    "feature_count":self.feature_count,"missing_features":list(row.missing_features),"backend":"xgboost_cpu_process",
                    "elapsed_ms":round((time.perf_counter()-started)*1000,3)},
                "stack":{**common,"model_id":"teammate/stacked_ensemble","raw_score":float(score),
                    "used_modalities":["tuned_dasheng","xgb_digest"],"not_used_modalities":["vision"],
                    "weights":dict(zip(manifest['branches'],manifest['weights'])),
                    "calibration_scope":"parameters_fit_on_courtroom_data;webcam_calibration_unverified",
                    **self._stack_fields(manifest),"caveats":manifest.get('caveats',[])}}

    def snapshot(self):
        with self._lifecycle_lock:
            process=self._process
            alive=process is not None and process.poll() is None
            ready=not self._closed and alive and self.manifest is not None
            return {"status":"closed" if self._closed else "ready" if ready else "unprepared",
                    "backend":"xgboost_cpu_process","worker_pid":process.pid if alive else None,
                    "alive":alive,"ready":ready,"feature_version":self.feature_version,"feature_count":self.feature_count}

    def close(self):
        with self._lifecycle_lock:
            self._closed=True;process=self._process;self._process=None
            self.model=None;self.manifest=None;self._worker_info=None
        self._terminate(process)
