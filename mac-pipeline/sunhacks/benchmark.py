"""Repeatable base model smoke benchmark on transient synthetic non-person inputs."""
import gc
import platform
try:
    import resource
except ImportError:
    resource = None
import time
import numpy as np
from .models import ModelRegistry, _sync

def benchmark_model(config,branch,repeats=2,frames=64):
    import torch
    registry = ModelRegistry(config)
    report = {"benchmark_type":"official_base_model_synthetic_smoke", "branch":branch,
              "platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,
              "mps_available":torch.backends.mps.is_available(),"trials":[],
              "input":"Transient generated color/motion pattern and sine wave; no people or dataset.",
              "capture_fps":None,"sample_fps":config.get("sample_fps",24),
              "frames_per_window":frames,"clip_span_seconds":(frames-1)/config.get("sample_fps",24),
              "quality_compared_to_64_frames":False,
              "reduced_frame_experiment":frames!=64}
    clip = np.zeros((frames,384,384,3),np.uint8)
    for i in range(frames): clip[i,:,i*4:i*4+16,1] = 160
    audio = (.02*np.sin(2*np.pi*220*np.arange(42000)/16000)).astype(np.float32)
    transcript = "This is a local pipeline smoke test with no claim about a person's behavior."
    started = time.perf_counter()
    try:
        model, processor, device, dtype, trained = registry._load(branch)
        _sync(torch,device)
        report.update(device=device,dtype=str(dtype),load_ms=(time.perf_counter()-started)*1000,
                      status="loaded",model_parameters=sum(p.numel() for p in model.parameters()) if hasattr(model,"parameters") else None)
        for n in range(max(1,min(5,repeats))):
            _sync(torch,device)
            started = time.perf_counter()
            vector,trained,device = registry._infer(branch,clip,audio,transcript)
            _sync(torch,device)
            seconds = time.perf_counter()-started
            report["trials"].append({"trial":n,"inference_and_preprocessing_ms":seconds*1000,
                                     "embedding_shape":list(vector.shape),"finite":bool(np.isfinite(vector).all()),
                                     "clip_frames_per_compute_second":frames/seconds if branch=="vision" else None,
                                     "fresh_inference_updates_per_second":1/seconds})
            report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed",error=type(exc).__name__+": "+str(exc)[:1000])
    finally:
        if resource is not None:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report["process_peak_rss_mb"] = rss/(1024*1024 if platform.system()=="Darwin" else 1024)
        else:
            import psutil
            info = psutil.Process().memory_info()
            report["process_peak_rss_mb"] = getattr(info,"peak_wset",info.rss)/(1024*1024)
        if torch.backends.mps.is_available():
            report["mps_current_allocated_mb"] = torch.mps.current_allocated_memory()/(1024*1024)
            report["mps_driver_allocated_mb"] = torch.mps.driver_allocated_memory()/(1024*1024)
        if torch.cuda.is_available():
            report["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated()/(1024*1024)
            report["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved()/(1024*1024)
        registry._release(branch)
    report["limitation"] = "Single-branch smoke measurement includes preprocessing; not simultaneous full-pipeline latency, sensor fps, classification quality or sustained thermal performance."
    return report
