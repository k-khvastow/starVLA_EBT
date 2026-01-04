import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple

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
        Calculates Scalar Energy for action sequence given layer-wise VLM context.
        """
        x = self.action_emb(actions) + self.pos_emb[:, :actions.shape[1], :]
        x = self.drop(x)
        
        for i, layer in enumerate(self.layers):
            # Select VLM layer (aligns with EBT depth)
            vlm_feats = cond_list[i]
            context = self.obs_emb_proj(vlm_feats)
            x = layer(x, context)
            
        x = self.norm(x)
        energies = self.energy_head(x).squeeze(-1)
        
        # Return total energy (scalar) for gradient computation
        # Using sum() preserves gradient magnitude better for optimization than mean()
        return energies.sum() 

class LayerwiseEBTActionHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        
        self.action_dim = config.action_dim
        self.horizon = config.future_action_window_size + 1
        
        # --- MCMC & Training Params ---
        # Number of steps to run optimization loop during training
        self.train_inference_steps = getattr(config, 'train_inference_steps', 10) 
        self.inference_steps = getattr(config, 'inference_steps', 30)
        
        # Gradient Descent Step Size (Alpha) - Learnable
        init_alpha = getattr(config, 'mcmc_step_size', 0.01)
        self.alpha = nn.Parameter(torch.tensor(init_alpha), requires_grad=True)
        
        # Langevin Noise
        init_noise = getattr(config, 'langevin_noise_std', 0.001)
        self.langevin_noise_std = nn.Parameter(torch.tensor(init_noise), requires_grad=True)

        # Constraints
        self.clip_grad_value = getattr(config, 'clip_grad_value', 1.0)
        self.clamp_actions_value = getattr(config, 'clamp_actions_value', 1.0) # Assume normalized actions [-1, 1]

        self.model = TransformerForEBT_Layerwise(
            action_dim=self.action_dim,
            cond_dim=full_config.framework.qwenvl.vl_hidden_dim,
            horizon=self.horizon,
            n_layer=getattr(config, 'ebt_layers', 6),
            n_head=getattr(config, 'ebt_heads', 8),
            n_emb=getattr(config, 'ebt_emb_dim', 512),
        )

    def _mcmc_step(self, actions, cond_list, create_graph=False, add_noise=True):
        """
        Perform one MCMC gradient descent step to minimize energy.
        IMPORTANT: create_graph=True allows differentiating through this step.
        """
        with torch.enable_grad():
            # Ensure actions require grad for energy computation
            actions = actions.detach().requires_grad_(True)
            
            # 1. Add Langevin Noise (Exploration)
            if add_noise and self.langevin_noise_std > 0:
                noise = torch.randn_like(actions) * self.langevin_noise_std
                actions_noisy = actions + noise
            else:
                actions_noisy = actions

            # 2. Compute Energy
            energy_sum = self.model(actions_noisy, cond_list)
            
            # 3. Compute Gradient of Energy w.r.t Actions
            # We want to MINIMIZE energy, so we move opposite to gradient
            grad = torch.autograd.grad(
                outputs=energy_sum,
                inputs=actions,
                create_graph=create_graph # Key for backprop through optimization
            )[0]
            
            # 4. Clip Gradient
            if self.clip_grad_value > 0:
                grad = torch.clamp(grad, -self.clip_grad_value, self.clip_grad_value)
            
            # 5. Update Actions (Gradient Descent)
            alpha = torch.clamp(self.alpha, min=1e-5, max=1.0)
            actions_new = actions - alpha * grad
            
            # 6. Clamp Actions
            if self.clamp_actions_value > 0:
                actions_new = torch.clamp(actions_new, -self.clamp_actions_value, self.clamp_actions_value)
                
            return actions_new

    def forward(self, vl_embs_list: List[torch.Tensor], actions: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        Training Forward Pass (Implicit Differentiation / Unrolled Optimization).
        
        1. Initialize random actions (Noise).
        2. Refine them using the model's energy function (Gradient Descent).
        3. Loss = MSE(Refined Actions, Ground Truth Actions).
        
        This forces the energy manifold to be shaped such that GD leads to GT.
        """
        # Align Layer Depth
        if len(vl_embs_list) > len(self.model.layers):
            vl_embs_list = vl_embs_list[-len(self.model.layers):]
            
        B, T, D = actions.shape
        device = actions.device
        
        # 1. Initialize corrupted/random actions
        # The provided code initializes from pure Gaussian noise
        predicted_actions = torch.randn((B, T, D), device=device, dtype=actions.dtype)
        
        # 2. MCMC Refinement Loop
        for step in range(self.train_inference_steps):
            # Crucial: Only create graph on the LAST step to save memory, 
            # unless you want full unrolling (expensive). 
            # The provided code snippet uses `create_graph` only on the last step.
            is_last_step = (step == self.train_inference_steps - 1)
            
            predicted_actions = self._mcmc_step(
                predicted_actions, 
                vl_embs_list, 
                create_graph=is_last_step, 
                add_noise=True
            )
            
            # If not last step, detach to prevent massive graph build-up
            if not is_last_step:
                predicted_actions = predicted_actions.detach()

        # 3. Compute Reconstruction Loss
        # Force the "Stable State" of the energy model to match Ground Truth
        loss = F.mse_loss(predicted_actions, actions)
        
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs_list: List[torch.Tensor], state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inference: Run MCMC without building graph.
        """
        if len(vl_embs_list) > len(self.model.layers):
            vl_embs_list = vl_embs_list[-len(self.model.layers):]
            
        B = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        
        # Initialize from noise
        predicted_actions = torch.randn((B, self.horizon, self.action_dim), device=device)
        
        # Run refinement
        for step in range(self.inference_steps):
            predicted_actions = self._mcmc_step(
                predicted_actions, 
                vl_embs_list, 
                create_graph=False, 
                add_noise=(self.langevin_noise_std > 0)
            )
            
        return predicted_actions

def get_action_model(config=None):
    return LayerwiseEBTActionHead(full_config=config)