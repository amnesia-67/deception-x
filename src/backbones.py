"""The five Branch-1 backbones from Gemini.pdf (+ V-JEPA 2.1), loaded from verified local weights.

Every wrapper maps raw inputs already on the GPU to a temporal token sequence
(B, 32, D) (text: (B, D)), matching the PDF's T=32 alignment:

  vjepa2     facebook/vjepa2-vitl-fpc64-256  64x256x256 -> 32 tubelets x 1024 (skip_predictor=True)
  vjepa2_1   V-JEPA 2.1 ViT-B/16 @384        32x384x384 -> 16 tubelets x 768 -> interp to 32
  dasheng    mispeech/dasheng-base           16 kHz wav -> 25 Hz x 768 -> interp to 32
  emotion2vec emotion2vec_plus_large          16 kHz wav -> 50 Hz x 1024 -> interp to 32
  qwen3      Qwen/Qwen3-Embedding-0.6B       instruction-prefixed transcript -> [EOS] -> MRL 128, L2

Base weights are frozen in bf16. LoRA adapters (fp32) go on the attention
projections of the last ``lora_blocks`` blocks only, so no autograd graph is
built for the frozen trunk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
T_OUT = 32
QWEN_INSTRUCTION = (
    "Given a verbatim spoken testimony transcript, represent its psycholinguistic, "
    "disfluency and reality-monitoring content for deception analysis"
)


@dataclass
class LoraSpec:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    blocks: int = 4  # last N transformer blocks get adapters


def interp_time(x: torch.Tensor, steps: int = T_OUT) -> torch.Tensor:
    """(B, T, D) -> (B, steps, D).

    Upsampling (T < steps) uses linear interpolation as the PDF specifies. Downsampling
    (e.g. 3000 emotion2vec frames -> 32) uses adaptive average pooling: linear interpolation
    would read only 2 neighbouring frames per output step and discard ~98% of the audio.
    """
    T = x.shape[1]
    if T == steps:
        return x
    xt = x.transpose(1, 2).float()
    if T > steps:
        y = F.adaptive_avg_pool1d(xt, steps)
    else:
        y = F.interpolate(xt, size=steps, mode="linear", align_corners=False)
    return y.transpose(1, 2).to(x.dtype)


def normalize_video(video_u8: torch.Tensor, res: int) -> torch.Tensor:
    """(B, T, H, W, 3) uint8 on GPU -> (B, T, 3, res, res) float, ImageNet-normalized."""
    B, T, H, W, _ = video_u8.shape
    x = video_u8.permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W).float().div_(255.0)
    if H != res or W != res:
        x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
    mean = x.new_tensor(IMNET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMNET_STD).view(1, 3, 1, 1)
    return ((x - mean) / std).view(B, T, 3, res, res)


def _apply_lora(model: nn.Module, target_regex: str, spec: LoraSpec) -> nn.Module:
    from peft import LoraConfig, inject_adapter_in_model

    cfg = LoraConfig(r=spec.r, lora_alpha=spec.alpha, lora_dropout=spec.dropout,
                     target_modules=target_regex, bias="none")
    model = inject_adapter_in_model(cfg, model)
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
        if "lora_" in n:
            p.data = p.data.float()
    return model


def _block_regex(prefix: str, first: int, last: int, leaves: str) -> str:
    idx = "|".join(str(i) for i in range(first, last + 1))
    return rf"{prefix}\.({idx})\.{leaves}"


class Backbone(nn.Module):
    name: str = "base"
    modality: str = "video"
    out_dim: int = 0

    def train(self, mode: bool = True):
        """Frozen trunk always in eval (no dropout noise / BN updates in frozen layers);
        only the LoRA dropout modules follow `mode`."""
        super().train(mode)
        self.model.eval()
        if mode:
            for n, m in self.model.named_modules():
                if "lora_dropout" in n:
                    m.train()
        return self

    def lora_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if p.requires_grad]

    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if "lora_" in k}

    def load_lora_state_dict(self, sd: Dict[str, torch.Tensor]):
        missing = [k for k in sd if k not in self.state_dict()]
        if missing:
            raise KeyError(f"unknown LoRA keys: {missing[:3]}")
        self.load_state_dict(sd, strict=False)


class VJEPA2Video(Backbone):
    name, modality, out_dim = "vjepa2", "video", 1024
    n_frames, res = 64, 256

    def __init__(self, lora: Optional[LoraSpec] = None, device="cuda"):
        super().__init__()
        from transformers import VJEPA2Model

        m = VJEPA2Model.from_pretrained(os.path.join(MODELS, "vjepa2-vitl-fpc64-256"),
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
        del m.predictor  # never used: forward runs with skip_predictor=True (PDF constraint #1)
        m.predictor = None
        for p in m.parameters():
            p.requires_grad = False
        n_layers = m.config.num_hidden_layers
        if lora is not None and lora.blocks > 0:
            m = _apply_lora(m, _block_regex(r"encoder\.layer", n_layers - lora.blocks, n_layers - 1,
                                            r"attention\.(query|key|value|proj)"), lora)
        self.model = m.to(device)
        self.grid = (self.n_frames // 2, self.res // 16, self.res // 16)

    def forward(self, video_u8: torch.Tensor) -> torch.Tensor:
        x = normalize_video(video_u8, self.res).to(torch.bfloat16)
        B = x.shape[0]
        # Encoder only == VJEPA2Model.forward(..., skip_predictor=True).last_hidden_state, minus the
        # unused full-size apply_masks() gather that forward() performs even when the predictor is skipped.
        out = self.model.encoder(pixel_values_videos=x).last_hidden_state
        t, h, w = self.grid
        return out.view(B, t, h * w, -1).mean(2)  # spatial pool -> (B, 32, 1024)


class VJEPA21Video(Backbone):
    """V-JEPA 2.1 ViT-B/16 @384 (HISTORICAL: kept to reproduce earlier results; never served)."""
    name, modality, out_dim = "vjepa2_1", "video", 768
    n_frames, res = 32, 384
    arch, weights_dir, depth = "vit_base", "vjepa2_1-vitb-384", 12

    def __init__(self, lora: Optional[LoraSpec] = None, device="cuda"):
        super().__init__()
        from safetensors.torch import load_file

        from third_party.vjepa2_1 import vision_transformer as vt

        # Encoder kwargs exactly as Meta's hub builder (_make_vjepa2_1_model) at the vendored commit.
        enc = getattr(vt, self.arch)(img_size=(self.res, self.res), num_frames=64, tubelet_size=2, patch_size=16,
                                     use_sdpa=True, use_silu=False, wide_silu=True, uniform_power=False,
                                     use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)
        if enc.embed_dim != self.out_dim or len(enc.blocks) != self.depth:
            raise ValueError(f"{self.arch} is not {self.out_dim} wide x {self.depth} blocks")
        res = enc.load_state_dict(load_file(os.path.join(MODELS, self.weights_dir, "encoder.safetensors")),
                                  strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        enc = enc.to(torch.bfloat16)
        for p in enc.parameters():
            p.requires_grad = False
        if lora is not None and lora.blocks > 0:
            n = len(enc.blocks)
            enc = _apply_lora(enc, _block_regex(r"blocks", n - lora.blocks, n - 1, r"attn\.(qkv|proj)"), lora)
        self.model = enc.to(device)
        self.grid = (self.n_frames // 2, self.res // 16, self.res // 16)

    def forward(self, video_u8: torch.Tensor) -> torch.Tensor:
        x = normalize_video(video_u8, self.res).to(torch.bfloat16)  # (B, T, 3, H, W)
        B = x.shape[0]
        out = self.model(x.permute(0, 2, 1, 3, 4).contiguous())  # (B, N, 768)
        t, h, w = self.grid
        return interp_time(out.view(B, t, h * w, -1).mean(2))


class VJEPA21LVideo(VJEPA21Video):
    """V-JEPA 2.1 ViT-L/16 @384 - Meta `vjepa2_1_vit_large_384` (ema_encoder): 1024 wide, 24 blocks, 16 heads.

    Uses the OFFICIAL L profile of Meta's hub builder (num_frames=64, tubelet 2, 384 px, patch 16), not the
    shorter 32-frame B route: 64 contiguous cached frames (stride 1) -> 32 native tubelets x 24x24 patches,
    spatially mean-pooled -> exactly the PDF's 32 steps, so no temporal interpolation is applied."""
    name, out_dim = "vjepa2_1_l", 1024
    n_frames = 64
    arch, weights_dir, depth = "vit_large", "vjepa2_1-vitl-384", 24


class DashengAudio(Backbone):
    name, modality, out_dim = "dasheng", "audio", 768

    def __init__(self, lora: Optional[LoraSpec] = None, device="cuda"):
        super().__init__()
        import torchaudio.transforms as AT
        from transformers import AutoModel

        m = AutoModel.from_pretrained(os.path.join(MODELS, "dasheng-base"), trust_remote_code=True)
        for p in m.parameters():
            p.requires_grad = False
        m = m.to(torch.bfloat16)
        # init_bn (BatchNorm over mel bins) must keep frozen running stats -> keep fp32 + eval
        m.encoder.init_bn.float()
        if lora is not None and lora.blocks > 0:
            n = len(m.encoder.blocks)
            m = _apply_lora(m, _block_regex(r"encoder\.blocks", n - lora.blocks, n - 1, r"attn\.(qkv|proj)"), lora)
        self.model = m.to(device)
        # identical front-end to models/dasheng-base/feature_extraction_dasheng.py
        self.mel = AT.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512, hop_length=160,
                                     f_min=0, f_max=8000, n_mels=64, center=True).to(device)
        self.to_db = AT.AmplitudeToDB(top_db=120).to(device)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            mel = self.to_db(self.mel(wav.float()))  # (B, 64, frames)
            x = self.model.encoder.init_bn(mel.unsqueeze(1))  # (B, 1, 64, frames); init_bn rearranges itself
        enc = self.model.encoder
        bn, enc.init_bn = enc.init_bn, None  # already applied above in fp32
        try:
            h = enc(x.to(enc.patch_embed.proj.weight.dtype))  # (B, T_d, 768) @ 25 Hz; bf16 served, fp32 reference
        finally:
            enc.init_bn = bn
        return interp_time(h)


class Emotion2vecAudio(Backbone):
    name, modality, out_dim = "emotion2vec", "audio", 1024
    EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

    def __init__(self, lora: Optional[LoraSpec] = None, device="cuda"):
        super().__init__()
        import yaml
        from safetensors.torch import load_file

        from third_party.emotion2vec.model import Emotion2vec

        d = os.path.join(MODELS, "emotion2vec_plus_large")
        cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))
        m = Emotion2vec(model_conf=cfg["model_conf"], vocab_size=9)
        sd = {k.replace("d2v_model.", "", 1): v for k, v in load_file(os.path.join(d, "model.safetensors")).items()}
        res = m.load_state_dict(sd, strict=False)
        bad_missing = [k for k in res.missing_keys if ".decoder." not in k]  # MAE decoder is training-only
        assert not bad_missing and not res.unexpected_keys, (bad_missing[:5], res.unexpected_keys[:5])
        for p in m.parameters():
            p.requires_grad = False
        m = m.to(torch.bfloat16)
        if lora is not None and lora.blocks > 0:
            n = len(m.blocks)
            m = _apply_lora(m, rf"blocks\.({'|'.join(str(i) for i in range(n - lora.blocks, n))})\.attn\.(qkv|proj)",
                            lora)
        self.model = m.to(device)

    def frames(self, wav: torch.Tensor) -> torch.Tensor:
        src = F.layer_norm(wav.float(), wav.shape[-1:])  # cfg.normalize=True (per-utterance)
        dt = next(self.model.blocks.parameters()).dtype  # bf16 served / trained; fp32 for the parity reference
        return self.model.extract_features(src.to(dt), padding_mask=None)["x"]  # (B, T_e, 1024)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return interp_time(self.frames(wav))

    @torch.no_grad()
    def emotion_probs(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.frames(wav).float().mean(1)
        return torch.softmax(self.model.proj.float()(x), -1)


class Qwen3Text(Backbone):
    name, modality, out_dim = "qwen3", "text", 128
    max_tokens = 512

    def __init__(self, lora: Optional[LoraSpec] = None, device="cuda", mrl_dim: int = 128):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        d = os.path.join(MODELS, "qwen3-embedding-0.6b")
        self.tok = AutoTokenizer.from_pretrained(d, padding_side="left")
        m = AutoModel.from_pretrained(d, dtype=torch.bfloat16, attn_implementation="sdpa")
        for p in m.parameters():
            p.requires_grad = False
        if lora is not None and lora.blocks > 0:
            n = m.config.num_hidden_layers
            m = _apply_lora(m, _block_regex(r"layers", n - lora.blocks, n - 1, r"self_attn\.(q_proj|k_proj|v_proj|o_proj)"),
                            lora)
        self.model = m.to(device)
        self.mrl_dim = mrl_dim
        self.out_dim = mrl_dim
        self.device_ = device

    def format(self, texts: List[str]) -> List[str]:
        return [f"Instruct: {QWEN_INSTRUCTION}\nQuery: {t}" for t in texts]

    def embed_full(self, texts: List[str]) -> torch.Tensor:
        batch = self.tok(self.format(texts), padding=True, truncation=True, max_length=self.max_tokens,
                         return_tensors="pt").to(self.device_)
        h = self.model(**batch).last_hidden_state
        return h[:, -1]  # left padding -> last position is the final ([EOS]) token

    def forward(self, texts: List[str]) -> torch.Tensor:
        e = self.embed_full(texts)[:, : self.mrl_dim].float()
        return F.normalize(e, dim=-1)


BACKBONES = {
    "vjepa2": VJEPA2Video,
    "vjepa2_1": VJEPA21Video,
    "vjepa2_1_l": VJEPA21LVideo,
    "dasheng": DashengAudio,
    "emotion2vec": Emotion2vecAudio,
    "qwen3": Qwen3Text,
}


class AttnPoolHead(nn.Module):
    """LayerNorm -> learned-query attention pooling over time -> dropout -> logit."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.float())
        if x.dim() == 2:
            return self.out(self.drop(x)).squeeze(-1)
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = (w.unsqueeze(-1) * x).sum(1)
        return self.out(self.drop(pooled)).squeeze(-1)
