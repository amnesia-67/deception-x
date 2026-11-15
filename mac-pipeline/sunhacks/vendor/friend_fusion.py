"""
Branch 1: Surgical Multi-Encoder & Discrepancy-Guided Fusion
(ReDA + DARE + DMC + MMPDA)
As specified in Gemini.pdf Section 2 (Pages 20-26).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class TwoLayerMLP(nn.Module):
    """2-layer MLP with LayerNorm and GELU projecting to shared d=256."""
    def __init__(self, in_features: int, hidden_features: int = 256, out_features: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DAREModule(nn.Module):
    """
    Discrepancy-Aware Redundancy Elimination (DARE).
    Calculates instantaneous audio-visual differential affinity D_t^{va},
    temporal salience s_t, and dynamic quantile soft-masking (rho=50%).
    """
    def __init__(self, d: int = 256, rho: float = 0.50):
        super().__init__()
        self.d = d
        self.rho = rho
        # Salience projection w_s: 2*d -> 1
        self.w_s = nn.Linear(2 * d, 1)
        self.gamma_d = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        h_v: torch.Tensor, # (B, T, d)
        h_a: torch.Tensor, # (B, T, d)
        h_l: torch.Tensor  # (B, d)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = h_v.shape
        
        # 1. Instantaneous cosine discrepancy D_t^{va}
        # Norms along feature dim
        h_v_norm = F.normalize(h_v, p=2, dim=-1)
        h_a_norm = F.normalize(h_a, p=2, dim=-1)
        
        # Dot product across feature dimension
        cos_sim_va = (h_v_norm * h_a_norm).sum(dim=-1, keepdim=True) # (B, T, 1)
        D_t_va = 1.0 - cos_sim_va # (B, T, 1)
        
        # 2. Global discrepancies D^{vl} and D^{al}
        h_v_bar = h_v.mean(dim=1) # (B, d)
        h_a_bar = h_a.mean(dim=1) # (B, d)
        
        cos_vl = F.cosine_similarity(h_v_bar, h_l, dim=-1) # (B,)
        cos_al = F.cosine_similarity(h_a_bar, h_l, dim=-1) # (B,)
        D_vl = 1.0 - cos_vl # (B,)
        D_al = 1.0 - cos_al # (B,)
        D_va_mean = D_t_va.squeeze(-1).mean(dim=1) # (B,)
        
        # 3. Temporal salience score s_t
        cat_va = torch.cat([h_v, h_a], dim=-1) # (B, T, 2d)
        salience_logits = self.w_s(cat_va) + self.gamma_d * D_t_va # (B, T, 1)
        s_t = torch.sigmoid(salience_logits).squeeze(-1) # (B, T)
        
        # 4. Dynamic soft-masking: retain top rho=50% highest-salience tokens
        # Compute quantile per batch item (cast to float for torch.quantile support)
        q_val = torch.quantile(s_t.float(), 1.0 - self.rho, dim=-1, keepdim=True).to(s_t.dtype) # (B, 1)
        mask_binary = (s_t >= q_val).to(s_t.dtype) # (B, T)
        m_t = (mask_binary * s_t).unsqueeze(-1) # (B, T, 1)
        
        # Masked pooling
        denom = m_t.sum(dim=1) + 1e-6 # (B, 1)
        h_v_hat = ((m_t * h_v).sum(dim=1) / denom).to(h_v.dtype) # (B, d)
        h_a_hat = ((m_t * h_a).sum(dim=1) / denom).to(h_a.dtype) # (B, d)
        
        return h_v_hat, h_a_hat, D_va_mean, D_vl, D_al

class MultiKernelMMD(nn.Module):
    """
    Multi-Kernel Maximum Mean Discrepancy for progressive domain adaptation (MMPDA).
    Uses a mixture of RBF kernels.
    """
    def __init__(self, kernels: Optional[list] = None):
        super().__init__()
        self.kernels = kernels or [0.5, 1.0, 2.0, 4.0, 8.0]

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
        # x: (N, d), y: (M, d)
        # Cast to float32 for CUDA cdist kernel support (bf16 cdist is unsupported in PyTorch)
        x_f = x.float()
        y_f = y.float()
        dist = torch.cdist(x_f, y_f, p=2) ** 2
        return torch.exp(-gamma * dist).to(dtype=x.dtype)

    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        N = source_features.size(0)
        M = target_features.size(0)
        if N == 0 or M == 0:
            return torch.tensor(0.0, device=source_features.device)
            
        k_ss = 0.0
        k_tt = 0.0
        k_st = 0.0
        
        for k in self.kernels:
            gamma = 1.0 / (2.0 * (k ** 2))
            k_ss = k_ss + self._rbf_kernel(source_features, source_features, gamma)
            k_tt = k_tt + self._rbf_kernel(target_features, target_features, gamma)
            k_st = k_st + self._rbf_kernel(source_features, target_features, gamma)
            
        mmd = k_ss.sum() / (N * N) + k_tt.sum() / (M * M) - 2.0 * k_st.sum() / (N * M)
        return F.relu(mmd)

class Branch1MultimodalNetwork(nn.Module):
    """
    Full Branch 1 Network combining:
    - Modality Projectors (Vision, Audio, Text -> d=256)
    - DARE temporal salience filtering
    - ReDA dynamic interaction tensor + modality reliability weights
    - Unimodal and Multimodal prediction heads
    - DMC and MMPDA loss computation
    """
    def __init__(
        self,
        visual_dim: int = 1024,
        audio_dim: int = 768,
        text_dim: int = 128,
        d: int = 256,
        dare_rho: float = 0.50,
        delta_margin: float = 0.45,
        lambda_1: float = 0.25,
        lambda_2: float = 0.05,
        lambda_3: float = 0.30
    ):
        super().__init__()
        self.d = d
        self.delta_margin = delta_margin
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        
        # 1. Modality Projectors
        self.projector_v = TwoLayerMLP(visual_dim, d, d)
        self.projector_a = TwoLayerMLP(audio_dim, d, d)
        self.projector_l = TwoLayerMLP(text_dim, d, d)
        
        # 2. DARE module
        self.dare = DAREModule(d=d, rho=dare_rho)
        
        # 3. ReDA Cross-Modal Discrepancy MLP
        # Input: [|h_v - h_a| || |h_v - h_l| || |h_a - h_l| || (h_v * h_a)] -> dim = 4*d = 1024
        self.mlp_disc = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU()
        )
        
        # 4. Modality Reliability Weights W_r
        # Input: [h_v_hat || h_a_hat || h_l || u_disc] -> dim = 4*d = 1024 -> 3
        self.w_r = nn.Linear(4 * d, 3)
        
        # 5. Prediction Heads
        self.head_fused = nn.Linear(d, 1)
        self.head_v = nn.Linear(d, 1)
        self.head_a = nn.Linear(d, 1)
        self.head_l = nn.Linear(d, 1)
        
        # 6. Domain adaptation MMD
        self.mmd = MultiKernelMMD()

    def forward(
        self,
        F_v: torch.Tensor, # (B, 32, 1024)
        F_a: torch.Tensor, # (B, 32, audio_dim)
        f_t: torch.Tensor  # (B, 128)
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = F_v.shape
        
        # Flatten temporal dimensions for projection through 2-layer MLPs
        h_v = self.projector_v(F_v.reshape(B * T, -1)).reshape(B, T, self.d)
        h_a = self.projector_a(F_a.reshape(B * T, -1)).reshape(B, T, self.d)
        h_l = self.projector_l(f_t) # (B, d)
        
        # DARE Filtering
        h_v_hat, h_a_hat, D_va, D_vl, D_al = self.dare(h_v, h_a, h_l)
        
        # ReDA Interaction Tensor u_disc
        diff_va = torch.abs(h_v_hat - h_a_hat)
        diff_vl = torch.abs(h_v_hat - h_l)
        diff_al = torch.abs(h_a_hat - h_l)
        prod_va = h_v_hat * h_a_hat
        
        cat_disc = torch.cat([diff_va, diff_vl, diff_al, prod_va], dim=-1) # (B, 4d)
        u_disc = self.mlp_disc(cat_disc) # (B, d)
        
        # Modality Reliability Weights alpha = [alpha_v, alpha_a, alpha_l]
        cat_rel = torch.cat([h_v_hat, h_a_hat, h_l, u_disc], dim=-1) # (B, 4d)
        alpha = F.softmax(self.w_r(cat_rel), dim=-1) # (B, 3)
        
        alpha_v = alpha[:, 0:1] # (B, 1)
        alpha_a = alpha[:, 1:2] # (B, 1)
        alpha_l = alpha[:, 2:3] # (B, 1)
        
        # Fused representation z_fused
        z_fused = alpha_v * h_v_hat + alpha_a * h_a_hat + alpha_l * h_l + u_disc # (B, d)
        
        # Predictions
        logit_fused = self.head_fused(z_fused).squeeze(-1) # (B,)
        p_neural = torch.sigmoid(logit_fused)
        
        p_v = torch.sigmoid(self.head_v(h_v_hat).squeeze(-1))
        p_a = torch.sigmoid(self.head_a(h_a_hat).squeeze(-1))
        p_l = torch.sigmoid(self.head_l(h_l).squeeze(-1))
        
        return {
            "p_neural": p_neural,
            "logit_fused": logit_fused,
            "z_fused": z_fused,
            "p_v": p_v,
            "p_a": p_a,
            "p_l": p_l,
            "alpha": alpha,
            "D_va": D_va,
            "D_vl": D_vl,
            "D_al": D_al,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        target_z_fused: Optional[torch.Tensor] = None,
        target_p_neural: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the complete Branch 1 objective:
        L_Branch1 = L_BCE(P_Neural, y) + lambda1 * L_DMC + lambda2 * L_ent + lambda3 * L_MMPDA + 1e-4 * ||Theta||_2^2
        """
        p_neural = outputs["p_neural"]
        logit_fused = outputs["logit_fused"]
        y = labels.float()
        
        # 1. Binary Cross Entropy Loss
        l_bce = F.binary_cross_entropy_with_logits(logit_fused, y)
        
        # 2. Shannon Entropy Loss (maximization -> minimize alpha * log(alpha))
        alpha = outputs["alpha"]
        l_ent = (alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean()
        
        # 3. Distilled Modality Consistency (DMC) Loss
        D_va = outputs["D_va"]
        D_vl = outputs["D_vl"]
        D_al = outputs["D_al"]
        
        d_sum = D_va + D_vl + D_al
        d_max = torch.maximum(torch.maximum(D_va, D_vl), D_al)
        margin_term = F.relu(self.delta_margin - d_max)
        
        # Discrepancy alignment term
        l_dmc_discrepancy = ((1.0 - y) * d_sum + y * margin_term).mean()
        
        # KL Divergence from stop_grad(P_Neural) to unimodal p_m
        p_stopped = p_neural.detach().clamp(1e-6, 1.0 - 1e-6)
        kl_sum = 0.0
        for p_m in [outputs["p_v"], outputs["p_a"], outputs["p_l"]]:
            p_m_c = p_m.clamp(1e-6, 1.0 - 1e-6)
            # Binary KL: P*log(P/p) + (1-P)*log((1-P)/(1-p))
            kl_m = p_stopped * torch.log(p_stopped / p_m_c) + (1.0 - p_stopped) * torch.log((1.0 - p_stopped) / (1.0 - p_m_c))
            kl_sum = kl_sum + kl_m.mean()
            
        l_dmc = l_dmc_discrepancy + kl_sum
        
        # 4. MMPDA Domain Adaptation Loss
        l_mmpda = torch.tensor(0.0, device=p_neural.device)
        if target_z_fused is not None and target_p_neural is not None:
            mmd_feat = self.mmd(outputs["z_fused"], target_z_fused)
            dec_align = F.l1_loss(outputs["p_neural"].mean(), target_p_neural.mean())
            l_mmpda = mmd_feat + dec_align
            
        # Total Branch 1 Loss (L2 weight decay is handled by AdamW optimizer)
        l_total = (
            l_bce
            + self.lambda_1 * l_dmc
            + self.lambda_2 * l_ent
            + self.lambda_3 * l_mmpda
        )
        
        return {
            "loss_total": l_total,
            "loss_bce": l_bce,
            "loss_dmc": l_dmc,
            "loss_ent": l_ent,
            "loss_mmpda": l_mmpda,
        }
