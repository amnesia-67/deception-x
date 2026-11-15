import argparse
import json
import os
from pathlib import Path
import platform
import time

def main():
    parser = argparse.ArgumentParser(description="Local Mac multimodal research pipeline")
    parser.add_argument("command",choices=("serve","doctor","prepare","benchmark","authorize-camera","authorize-audio","fetch-rppg"),nargs="?",default="serve")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--env-file",type=Path,help="Explicit local API-key file; never executed as a shell script")
    parser.add_argument("--port",type=int,default=8765)
    parser.add_argument("--demo",action="store_true")
    parser.add_argument("--seconds",type=float,default=10)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--branch",choices=("vision","dasheng","emotion","text"))
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--frames",type=int,choices=(8,16,32,64),default=64,help="Benchmark only; reduced-frame quality is unverified")
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    env_file = args.env_file or config.get("env_file")
    if env_file:
        from .credentials import load_env_file
        if config.get("claim_check",{}).get("provider")=="openrouter":
            load_env_file(env_file,allowed_keys=("OPENROUTER_API_KEY",))
        else:
            load_env_file(env_file)
    # Conservative process allocation cap; never disable the MPS allocator guard.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO","0.55")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO","0.45")
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    threads = config.get("cpu_threads")
    if threads is not None:
        if isinstance(threads,bool) or not isinstance(threads,int) or not 1 <= threads <= 8:
            raise ValueError("cpu_threads must be an integer within1..8")
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    if args.command == "fetch-rppg":
        # Optional: pre-cache the display-only heart-rate weights before going offline.
        from .rppg import fetch_all
        _report(fetch_all(),args.output)
        return
    if args.command == "prepare":
        if config.get("runtime_mode") == "split_local_video":
            from .pipeline import Pipeline
            pipeline = Pipeline(config)
            try:
                pipeline.prepare_runtime()
                while pipeline.prepare_thread is not None and pipeline.prepare_thread.is_alive(): time.sleep(.1)
                _report(pipeline.state()["runtime"],args.output)
            finally: pipeline.close()
        else:
            if config.get("runtime_mode") != "remote_fast":parser.error("prepare requires a prepared-model config")
            from .fast_runtime import FastRuntime
            runtime = FastRuntime(config)
            try:_report(runtime.prepare(),args.output)
            finally:runtime.close()
    elif args.command.startswith("authorize-"):
        from .capture import camera_authorization_probe, microphone_authorization_probe
        _report(camera_authorization_probe(config.get("camera_device",0)) if args.command=="authorize-camera" else microphone_authorization_probe(config.get("audio_device")),args.output)
    elif args.command=="doctor":
        from .capture import probe_devices
        report = {"platform":platform.platform(),"python":platform.python_version(),"devices":probe_devices()}
        try:
            import torch
            report["torch"] = torch.__version__
            report["mps_available"] = torch.backends.mps.is_available()
        except ImportError: report["torch"] = "not installed"
        _report(report,args.output)
    elif args.command=="benchmark":
        if args.branch:
            from .benchmark import benchmark_model
            report = benchmark_model(config,args.branch,args.repeats,args.frames)
        else:
            from .pipeline import Pipeline
            pipeline = Pipeline(config)
            pipeline.start("demo")
            deadline = time.monotonic()+max(3,min(120,args.seconds))
            while time.monotonic()<deadline: time.sleep(.1)
            report = pipeline.state()
            pipeline.stop()
            report["benchmark_type"] = "synthetic_orchestration_only; not model throughput"
        _report(report,args.output)
    else:
        # Load native capture libraries on the process main thread. Device
        # streams remain closed until an explicit live-session request.
        import importlib
        for module in ("cv2","sounddevice"):
            try: importlib.import_module(module)
            except ImportError: pass
        from .pipeline import Pipeline
        from .server import make_server
        pipeline = Pipeline(config)
        server = make_server(pipeline,args.port)
        if (config.get("fast_runtime",{}).get("prepare_on_start") or
                config.get("split_runtime",{}).get("prepare_on_start")):
            pipeline.prepare_runtime()
        if args.demo: pipeline.start("demo")
        print(f"SunHacks local dashboard: http://127.0.0.1:{server.server_port}",flush=True)
        print("Sensors are off until explicitly started. Ctrl-C stops capture.",flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
        finally: pipeline.close(); server.server_close()

def _report(report,path):
    from .pipeline import public
    data = json.dumps(public(report),indent=2,allow_nan=False)
    if path:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(data+"\n")
    print(data)

if __name__=="__main__": main()
