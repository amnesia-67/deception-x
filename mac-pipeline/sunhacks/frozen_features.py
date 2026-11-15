"""Resident frozen Qwen/emotion features for the reviewed real 89-field digest.

All inference is local. The Qwen instruction/prototypes and emotion probability
head reproduce the new ensemble extraction; pooled emotion features are not
substituted for class probabilities. Static intent preparation is a real local
forward, with no participant data. Precision parity to GPU bf16 is unverified.
"""
from __future__ import annotations
import copy
import threading
import time
import numpy as np

from .local_digest import (EMOTIONS, INTENTS, QWEN_INTENT_MAX_TOKENS,
                           QWEN_TESTIMONY_MAX_TOKENS, format_testimony)
from .models import MODEL_IDS, ModelRegistry


def eval_model_audio(audio):
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1 or len(values) < 8000 or not np.isfinite(values).all():
        raise ValueError("Need >=0.5s complete finite mono16kHz audio")
    if len(values) > 60 * 16000:
        start = (len(values) - 60 * 16000) // 2
        values = values[start:start + 60 * 16000]
    if len(values) < 16000:
        values = np.pad(values, (0, 16000 - len(values)))
    return np.ascontiguousarray(values, dtype=np.float32)


class FrozenFeatureModels:
    def __init__(self, config, *, registry=None):
        self.config = copy.deepcopy(config)
        for name in ("text", "emotion"):
            cfg = self.config.setdefault("models", {}).setdefault(name, {})
            if cfg.get("adapter_path"):
                raise ValueError("The real digest requires frozen Qwen/emotion bases, not their tuned adapters")
            cfg["keep_loaded"] = True
        self.registry = registry if registry is not None else ModelRegistry(self.config)
        self.intent_embeddings = None
        self._lock = threading.RLock()
        self._closed = False

    @staticmethod
    def _sync(torch, device):
        if str(device).startswith("mps"):
            torch.mps.synchronize()
        elif str(device).startswith("cuda"):
            torch.cuda.synchronize()

    def _qwen(self, texts, *, max_tokens):
        import torch
        model, tokenizer, device, dtype, trained = self.registry._load("text")
        if trained:
            raise ValueError("Digest Qwen features must use the frozen base")
        model.eval().requires_grad_(False)
        inputs = tokenizer(list(texts), padding=True, truncation=True,
                           max_length=max_tokens, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        # Left padding is part of the reviewed Qwen contract.
        if not bool((inputs["attention_mask"][:, -1] == 1).all()):
            raise ValueError("Qwen final-token pooling requires left-padded nonempty inputs")
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state[:, -1].float()
        self._sync(torch, device)
        result = hidden.cpu().numpy()
        if result.shape != (len(texts), 1024) or not np.isfinite(result).all():
            raise ValueError("Frozen Qwen emitted invalid full1024 features")
        return np.ascontiguousarray(result), str(device), str(dtype)

    def prepare(self):
        started = time.perf_counter()
        with self._lock:
            if self._closed:
                return {"status":"stopped", "components":{}}
            components = {}
            for name in ("text", "emotion"):
                try:
                    self.registry._load(name)
                    components[name] = {"status":"ready", "weights_loaded":True,
                                        "model_id":MODEL_IDS[name], "kernels_warmed":False,
                                        "tuning":"frozen_base_for_digest"}
                except Exception as exc:
                    components[name] = {"status":"unavailable", "reason":type(exc).__name__+": "+str(exc)[:300]}
            if components["text"]["status"] == "ready" and self.intent_embeddings is not None:
                components["text"].update(kernels_warmed=True,static_intents_prepared=33,intent_cache_hit=True)
            elif components["text"]["status"] == "ready":
                try:
                    # No instruction prefix on the 33 intent documents. Chunking
                    # bounds preparation memory; left-padding keeps final pooling.
                    all_rows = []
                    for start in range(0, len(INTENTS), 8):
                        row, _, _ = self._qwen(INTENTS[start:start+8], max_tokens=QWEN_INTENT_MAX_TOKENS)
                        all_rows.append(row)
                    array = np.concatenate(all_rows)
                    norms = np.linalg.norm(array, axis=1, keepdims=True)
                    if np.any(norms <= 0):
                        raise ValueError("Zero frozen intent embedding")
                    self.intent_embeddings = np.ascontiguousarray(array/norms, dtype=np.float32)
                    components["text"].update(kernels_warmed=True, static_intents_prepared=33)
                except Exception as exc:
                    components["text"].update(status="unavailable", reason=type(exc).__name__+": "+str(exc)[:300])
            return {"status":"ready" if all(v["status"]=="ready" for v in components.values()) else "partial",
                    "components":components,"elapsed_ms":round((time.perf_counter()-started)*1000,3),
                    "participant_data_used":False,"network_used":False}

    def text(self, transcript):
        started = time.perf_counter()
        with self._lock:
            if self._closed:
                raise RuntimeError("Frozen feature models are closed")
            if self.intent_embeddings is None:
                raise RuntimeError("Frozen intent prototypes have not been prepared")
            full, device, dtype = self._qwen([format_testimony(transcript)], max_tokens=QWEN_TESTIMONY_MAX_TOKENS)
            mrl = full[0,:128].copy()
            norm = float(np.linalg.norm(mrl))
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("Qwen first128 embedding cannot be normalized")
            mrl /= norm
            return {"status":"base", "model_id":MODEL_IDS["text"],"tuning":"frozen_base_for_digest",
                    "device":device,"dtype":dtype,"embedding_shape":[128],"pooling":"last_token_full1024_then_mrl128_l2",
                    "_embedding":mrl,"_full_embedding":full[0],"_intent_embeddings":self.intent_embeddings,
                    "elapsed_ms":round((time.perf_counter()-started)*1000,3),"locality":"mac",
                    "precision_parity_verified":False,"reason":"Frozen instructed Qwen features and33 raw-document intent prototypes; no tuned Qwen adapter is used by this digest."}

    def emotion(self, audio):
        started = time.perf_counter()
        import torch
        with self._lock:
            if self._closed:
                raise RuntimeError("Frozen feature models are closed")
            wrapper, _, device, dtype, trained = self.registry._load("emotion")
            if trained:
                raise ValueError("Digest emotion features require the frozen base")
            model = wrapper.model
            model.eval().requires_grad_(False)
            values = eval_model_audio(audio)
            waveform = torch.from_numpy(values).to(device).unsqueeze(0)
            waveform = torch.nn.functional.layer_norm(waveform.float(), waveform.shape[-1:])
            with torch.inference_mode():
                sequence = model.extract_features(waveform.to(dtype), padding_mask=None)["x"].float()
                pooled = sequence.mean(1)
                probabilities = torch.softmax(model.proj.float()(pooled), dim=-1)
            self._sync(torch, device)
            probs = probabilities[0].cpu().numpy()
            vector = pooled[0].cpu().numpy()
            if (probs.shape!=(9,) or vector.shape!=(1024,) or not np.isfinite(probs).all() or
                    not np.isfinite(vector).all() or not np.isclose(probs.sum(),1,atol=1e-4)):
                raise ValueError("Frozen emotion2vec emitted invalid nine-class probabilities")
            return {"status":"base","model_id":MODEL_IDS["emotion"],"tuning":"frozen_base_for_digest",
                    "device":str(device),"dtype":str(dtype),"embedding_shape":[1024],
                    "_embedding":vector,"_class_probabilities":probs,
                    "class_probabilities":dict(zip(EMOTIONS,map(float,probs))),
                    "model_audio_samples":len(values),"locality":"mac","precision_parity_verified":False,
                    "elapsed_ms":round((time.perf_counter()-started)*1000,3),
                    "reason":"Frozen nine-class emotion probabilities for the digest; no tuned emotion adapter is used by the exported stack."}

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.intent_embeddings = None
            self.registry.close()
