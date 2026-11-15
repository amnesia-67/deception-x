"""Standalone local XGBoost worker. Never import torch or the sunhacks package.

Run by local_stack_process.py with private stdio pipes. The only third-party
imports are NumPy and XGBoost, inside the validated prepare path. No features,
request bodies, environment values or tracebacks are logged.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

PROTOCOL = "sunhacks-xgb-worker-v1"
MAX_LINE_BYTES = 65536
MAX_MODEL_BYTES = 4 * 1024 * 1024
MAX_SCHEMA_BYTES = 65536
SCHEMA_CONTRACTS = {
    "v2-no-within-z": ("85779f9ddb30a42af7a7157642e34d7fcf7c5ebe58c35473ccc74932dcfd3bce", 89),
    "v3-fixed-lag-motion": ("bbed857c527031550a878e01e4be8670fcd2a169eea07a482389c4ead31b5356", 88),
}


def pairs(items):
    out={}
    for key,value in items:
        if key in out:raise ValueError("duplicate field")
        out[key]=value
    return out


def load_json(data):
    return json.loads(data.decode("utf-8"),object_pairs_hook=pairs,
                      parse_constant=lambda value:(_ for _ in ()).throw(ValueError("nonfinite JSON")))


def bounded_artifact(path,digest,limit):
    if not isinstance(path,str) or not isinstance(digest,str) or re.fullmatch(r"[0-9a-f]{64}",digest) is None:
        raise ValueError("invalid artifact pin")
    with Path(path).open("rb") as stream:data=stream.read(limit+1)
    if not data or len(data)>limit or hashlib.sha256(data).hexdigest()!=digest:
        raise ValueError("artifact mismatch")
    return data


def names_digest(names):
    return hashlib.sha256(json.dumps(names,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()


def schema_contract(request):
    """Bind the protocol identity to one reviewed schema/version/count tuple."""
    version=request.get("feature_version", "v2-no-within-z")
    if not isinstance(version,str) or version not in SCHEMA_CONTRACTS:
        raise ValueError("unknown feature schema version")
    digest,count=SCHEMA_CONTRACTS[version]
    requested_count=request.get("feature_count",89 if version=="v2-no-within-z" else None)
    if (request.get("schema_sha256")!=digest or type(requested_count) is not int or
            requested_count!=count):
        raise ValueError("feature schema version/hash/count mismatch")
    return version,count


def prepare(request):
    if request.get("protocol")!=PROTOCOL:raise ValueError("wrong protocol")
    feature_version,feature_count=schema_contract(request)
    model_bytes=bounded_artifact(request.get("model_path"),request.get("model_sha256"),MAX_MODEL_BYTES)
    schema_bytes=bounded_artifact(request.get("schema_path"),request.get("schema_sha256"),MAX_SCHEMA_BYTES)
    schema=load_json(schema_bytes)
    names=schema.get("feature_names")
    if (schema.get("feature_version")!=feature_version or not isinstance(names,list) or len(names)!=feature_count or
            any(not isinstance(name,str) or not name or len(name)>100 for name in names) or len(set(names))!=feature_count or
            names_digest(names)!=request.get("feature_names_sha256")):
        raise ValueError("wrong feature schema")
    # This child alone owns the XGBoost/OpenMP runtime. It never loads Torch.
    import numpy as np
    import xgboost as xgb
    if "torch" in sys.modules or "sunhacks" in sys.modules:
        raise RuntimeError("worker imported a model package")
    model=xgb.Booster(params={"device":"cpu","nthread":2})
    model.load_model(bytearray(model_bytes));model.set_param({"device":"cpu","nthread":2})
    config=json.loads(model.save_config())
    if (model.num_features()!=feature_count or (model.feature_names is not None and model.feature_names!=names) or
            config["learner"]["objective"]["name"]!="binary:logistic"):
        raise ValueError("wrong model contract")
    state={"model":model,"names":names,"numpy":np,"xgb":xgb,
           "feature_version":feature_version,"feature_count":feature_count}
    reply={"protocol":PROTOCOL,"status":"ready","pid":os.getpid(),"feature_count":feature_count,"feature_version":feature_version,
           "model_sha256":request["model_sha256"],"schema_sha256":request["schema_sha256"],
           "feature_names_sha256":names_digest(names),"torch_imported":False,
           "backend":"xgboost_cpu_process","xgboost_version":str(xgb.__version__)}
    return state,reply


def predict(state,request):
    if request.get("protocol")!=PROTOCOL:raise ValueError("wrong protocol")
    feature_version=state.get("feature_version","v2-no-within-z")
    if feature_version not in SCHEMA_CONTRACTS:raise ValueError("unknown prepared feature version")
    feature_count=state.get("feature_count",len(state["names"]))
    if type(feature_count) is not int or feature_count!=SCHEMA_CONTRACTS[feature_version][1] or len(state["names"])!=feature_count:
        raise ValueError("wrong prepared feature schema")
    request_version=request.get("feature_version","v2-no-within-z" if feature_version=="v2-no-within-z" else None)
    request_count=request.get("feature_count",89 if feature_version=="v2-no-within-z" else None)
    if request_version!=feature_version or type(request_count) is not int or request_count!=feature_count:
        raise ValueError("prediction feature schema identity mismatch")
    values=request.get("values")
    if not isinstance(values,list) or len(values)!=feature_count:raise ValueError("wrong row width")
    for value in values:
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):
            raise ValueError("invalid row value")
    if request.get("feature_names_sha256")!=names_digest(state["names"]):raise ValueError("wrong row order")
    np=state["numpy"];xgb=state["xgb"]
    row=np.asarray([[np.nan if value is None else value for value in values]],dtype=np.float32)
    if np.isinf(row).any():raise ValueError("row value overflows float32")
    result=np.asarray(state["model"].predict(xgb.DMatrix(row,feature_names=state["names"]))).reshape(-1)
    if result.shape!=(1,) or not np.isfinite(result).all() or not 0<=result[0]<=1:raise ValueError("invalid score")
    return {"protocol":PROTOCOL,"status":"success","raw_score":float(result[0]),
            "source_window_id":request.get("source_window_id"),
            "feature_version":feature_version,"feature_count":feature_count}


def main():
    state=None
    while True:
        line=sys.stdin.buffer.readline(MAX_LINE_BYTES+1)
        if not line:return 0
        request_id=None
        try:
            if len(line)>MAX_LINE_BYTES or not line.endswith(b"\n"):raise ValueError("invalid framing")
            request=load_json(line)
            if not isinstance(request,dict):raise ValueError("expected object")
            request_id=request.get("request_id")
            if not isinstance(request_id,str) or re.fullmatch(r"[0-9a-f]{32}",request_id) is None:
                raise ValueError("invalid request identity")
            action=request.get("action")
            if action=="prepare" and state is None:state,reply=prepare(request)
            elif action=="predict" and state is not None:reply=predict(state,request)
            else:raise ValueError("invalid worker action")
            reply["request_id"]=request_id
        except Exception as exc:
            # No exception text: native exceptions can include row contents.
            reply={"protocol":PROTOCOL,"status":"error","code":type(exc).__name__,"request_id":request_id}
        data=json.dumps(reply,allow_nan=False,separators=(",",":")).encode()+b"\n"
        if len(data)>MAX_LINE_BYTES:return 2
        try:sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
        except (BrokenPipeError,OSError):return 0
        if reply["status"]=="error":return 2


if __name__=="__main__":
    raise SystemExit(main())
