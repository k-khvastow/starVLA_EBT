import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, p_drop: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, p_drop: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(x)

class CrossAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, p_drop: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(B, context.shape[1], 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class EBTBlockWithCrossAttn(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, p_drop):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = SelfAttention(dim, n_heads, p_drop)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, n_heads, p_drop)
        self.norm3 = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, p_drop)

    def forward(self, x, context):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.ffn(self.norm3(x))
        return x

class TransformerForEBT_Layerwise(nn.Module):
    def __init__(self, action_dim, cond_dim, horizon, n_layer=6, n_head=8, n_emb=512):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_emb = n_emb
        
        self.action_emb = nn.Linear(action_dim, n_emb)
        self.drop = nn.Dropout(0.1)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        
        # Projector for VLM features
        self.obs_emb_proj = nn.Linear(cond_dim, n_emb)
        
        self.layers = nn.ModuleList([
            EBTBlockWithCrossAttn(n_emb, n_head, 4 * n_emb, 0.1)
            for _ in range(n_layer)
        ])
        
        self.norm = RMSNorm(n_emb)
        self.energy_head = nn.Linear(n_emb, 1, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, actions: torch.Tensor, cond_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculates Energy for action sequence given layer-wise VLM context.
        """
        x = self.action_emb(actions) + self.pos_emb[:, :actions.shape[1], :]
        x = self.drop(x)
        
        # Iterate through EBT layers, each attending to corresponding VLM layer
        for i, layer in enumerate(self.layers):
            # Pick corresponding VLM layer (assuming list is aligned or we take last N)
            vlm_feats = cond_list[i]
            context = self.obs_emb_proj(vlm_feats)
            x = layer(x, context)
            
        x = self.norm(x)
        energies = self.energy_head(x).squeeze(-1)
        
        # Return mean energy of the sequence
        return energies.mean(dim=1)


class LayerwiseEBTActionHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        
        self.action_dim = config.action_dim
        self.horizon = config.future_action_window_size + 1
        
        # Training / MCMC Params
        self.train_n_mcmc = getattr(config, 'train_n_mcmc', 10)
        self.inference_n_mcmc = getattr(config, 'inference_n_mcmc', 30)
        self.mcmc_noise_scale = getattr(config, 'mcmc_noise_scale', 0.01)
        self.energy_reg_weight = getattr(config, 'energy_reg_weight', 1.0)
        
        # Learnable Step Size
        self.alpha = nn.Parameter(torch.tensor(0.01), requires_grad=True)

        self.model = TransformerForEBT_Layerwise(
            action_dim=self.action_dim,
            cond_dim=full_config.framework.qwenvl.vl_hidden_dim,
            horizon=self.horizon,
            n_layer=getattr(config, 'ebt_layers', 6),
            n_head=getattr(config, 'ebt_heads', 8),
            n_emb=getattr(config, 'ebt_emb_dim', 512),
        )

    def langevin_sampler(self, current_actions, cond_list, n_steps, step_size, noise_scale, train_mode=False):
        """
        Samples low-energy actions (Negatives) via Langevin Dynamics.
        """
        actions = current_actions.detach().clone()
        actions.requires_grad_(True)
        
        for _ in range(n_steps):
            energy = self.model(actions, cond_list)
            # Minimize energy (gradient descent)
            grad = torch.autograd.grad(energy.sum(), actions, create_graph=train_mode)[0]
            noise = torch.randn_like(actions) * noise_scale
            actions = actions - step_size * grad + noise
            
        return actions

    def forward(self, vl_embs_list: List[torch.Tensor], actions: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        1. Sample Negatives (Fantasies).
        2. Compute BCE Loss (Real vs Fake) + L2 Regularization.
        """
        # Align VLM layers to EBT layers
        if len(vl_embs_list) > len(self.model.layers):
            vl_embs_list = vl_embs_list[-len(self.model.layers):]
            
        # 1. Generate Negatives (Langevin Chains)
        initial_negatives = torch.randn_like(actions)
        negatives = self.langevin_sampler(
            initial_negatives, vl_embs_list, 
            n_steps=self.train_n_mcmc, step_size=self.alpha, noise_scale=self.mcmc_noise_scale
        )
        
        # 2. Compute Energies (Positive = Real, Negative = Fake)
        energy_pos = self.model(actions, vl_embs_list)
        energy_neg = self.model(negatives.detach(), vl_embs_list)
        
        # 3. Compute Loss
        # We treat (-Energy) as Logits.
        # Real Actions -> Label 1 (High Logit, Low Energy)
        # Fake Actions -> Label 0 (Low Logit, High Energy)
        
        cat_logits = torch.cat([-energy_pos, -energy_neg], dim=0)
        cat_labels = torch.cat([
            torch.ones_like(energy_pos), 
            torch.zeros_like(energy_neg)
        ], dim=0)
        
        # Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy_with_logits(cat_logits, cat_labels)
        
        # L2 Regularization on Energy Magnitudes
        reg_loss = self.energy_reg_weight * (energy_pos**2 + energy_neg**2).mean()
        
        return bce_loss + reg_loss

    @torch.no_grad()
    def predict_action(self, vl_embs_list: List[torch.Tensor], state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inference: Minimize Energy starting from noise.
        """
        if len(vl_embs_list) > len(self.model.layers):
            vl_embs_list = vl_embs_list[-len(self.model.layers):]
            
        B = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        
        actions = torch.randn((B, self.horizon, self.action_dim), device=device)
        
        with torch.enable_grad():
             actions = self.langevin_sampler(
                actions, vl_embs_list, 
                n_steps=self.inference_n_mcmc, step_size=self.alpha.abs(), noise_scale=0.0
            )
        
        return actions.detach()

def get_action_model(config=None):
    return LayerwiseEBTActionHead(full_config=config)