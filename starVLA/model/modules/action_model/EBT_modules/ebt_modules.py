"""
EBT Transformer Backbone
Ported from diffusion_policy/model/ebt/transformer_for_ebt.py
Adapted for starVLA by removing external dependencies.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class TransformerForEBT(nn.Module):
    """
    Transformer that outputs energy for Energy-Based Training (EBT).
    """
    
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 8,
        n_head: int = 8,
        n_emb: int = 256,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = True,
        obs_as_cond: bool = True,
        n_cond_layers: int = 2,
        energy_head_hidden_dim: int = None,
    ) -> None:
        super().__init__()

        # compute number of tokens
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.n_emb = n_emb
        self.obs_as_cond = obs_as_cond
        
        # Action embedding
        self.action_emb = nn.Linear(action_dim, n_emb)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # Observation conditioning
        self.cond_obs_emb = None
        self.cond_pos_emb = None
        self.obs_encoder = None
        self.action_decoder = None
        
        if obs_as_cond:
            assert cond_dim > 0, "cond_dim must be > 0 when obs_as_cond is True"
            
            # Observation embedding
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, n_obs_steps, n_emb))
            
            # Observation encoder
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.obs_encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                # Simple MLP encoder
                self.obs_encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            
            # Action decoder (conditioned on observations)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.action_decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # Encoder-only architecture (no observation conditioning)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.action_decoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # Causal attention masks
        if causal_attn:
            # Causal mask for action sequence
            sz = horizon
            mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
            mask = mask.float().masked_fill(mask, float('-inf'))
            self.register_buffer("action_mask", mask)
            
            if obs_as_cond:
                # Memory mask: actions can attend to all observations
                memory_mask = torch.zeros(horizon, n_obs_steps)
                self.register_buffer('memory_mask', memory_mask)
            else:
                self.memory_mask = None
        else:
            self.action_mask = None
            self.memory_mask = None

        # Energy head
        self.ln_f = nn.LayerNorm(n_emb)
        
        if energy_head_hidden_dim is None:
            energy_head_hidden_dim = n_emb
        
        # Energy head: pool action embeddings then predict scalar energy
        self.energy_head = nn.Sequential(
            nn.Linear(n_emb, energy_head_hidden_dim),
            nn.Mish(),
            nn.Linear(energy_head_hidden_dim, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential
        )
        
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                torch.nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                torch.nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForEBT):
            torch.nn.init.normal_(module.action_pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)

    def forward(
        self,
        actions: torch.Tensor,
        cond: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute energy for action sequence.
        
        Args:
            actions: (B, horizon, action_dim) - action sequence
            cond: (B, n_obs_steps, cond_dim) - observation features
        
        Returns:
            energy: (B,) - scalar energy for each batch element
        """
        
        # Embed actions
        action_emb = self.action_emb(actions)  # (B, horizon, n_emb)
        action_emb = action_emb + self.action_pos_emb
        action_emb = self.drop(action_emb)
        
        # Process through transformer
        if self.obs_as_cond and cond is not None:
            # Encode observations
            cond_emb = self.cond_obs_emb(cond)  # (B, n_obs_steps, n_emb)
            cond_emb = cond_emb + self.cond_pos_emb
            cond_emb = self.drop(cond_emb)
            
            # Encode observations
            if isinstance(self.obs_encoder, nn.TransformerEncoder):
                obs_memory = self.obs_encoder(cond_emb)
            else:
                obs_memory = self.obs_encoder(cond_emb)
            
            # Decode actions conditioned on observations
            x = self.action_decoder(
                tgt=action_emb,
                memory=obs_memory,
                tgt_mask=self.action_mask,
                memory_mask=self.memory_mask
            )
        else:
            # Encoder-only mode
            x = self.action_decoder(
                src=action_emb,
                mask=self.action_mask
            )
        
        # Compute energy
        x = self.ln_f(x)  # (B, horizon, n_emb)
        
        # Pool across sequence dimension (mean pooling)
        x = x.mean(dim=1)  # (B, n_emb)
        
        # Predict scalar energy
        energy = self.energy_head(x).squeeze(-1)  # (B,)
        
        return energy