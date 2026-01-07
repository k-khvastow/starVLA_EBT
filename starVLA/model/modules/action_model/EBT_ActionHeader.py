"""
EBT Action Head
Implements an Energy-Based Model using a Transformer backbone for action prediction.
Uses Differentiable Langevin Dynamics for training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from starVLA.model.modules.action_model.EBT_modules.ebt_modules import TransformerForEBT_Layerwise

class EBT_ActionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- 1. Extract Dimensions & Hyperparameters ---
        action_cfg = config.framework.action_model
        
        self.action_dim = action_cfg.action_dim
        # Input dim is the VLM hidden size (mapped from config)
        self.cond_dim = action_cfg.diffusion_model_cfg.cross_attention_dim 
        self.horizon = action_cfg.future_action_window_size + 1
        
        # EBT Specific Params (with defaults if not in config)
        self.n_emb = action_cfg.get('n_emb', 256)
        self.n_layer = action_cfg.get('n_layer', 4)
        self.n_head = action_cfg.get('n_head', 4)
        self.dropout = action_cfg.get('dropout', 0.1)
        
        # MCMC / Langevin Dynamics Params
        self.train_mcmc_steps = config.trainer.get('repeated_diffusion_steps', 4) 
        self.inf_mcmc_steps = action_cfg.get('inference_steps', 10)
        self.mcmc_noise_std = action_cfg.get('mcmc_noise_std', 0.001)
        initial_step_size = action_cfg.get('step_size', 0.01)
        
        # Max sequence length for VLM output conditioning (needed for EBT pos embeddings)
        self.max_cond_len = action_cfg.get('max_seq_len', 2048)

        # --- 2. Initialize Energy Function Model ---
        self.model = TransformerForEBT_Layerwise(
            action_dim=self.action_dim,
            cond_dim=self.cond_dim,
            horizon=self.horizon,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_emb=self.n_emb
        )
        
        # --- 3. Learnable Step Size (Alpha) ---
        # We make the Langevin step size learnable for better convergence
        self.alpha = nn.Parameter(torch.tensor(initial_step_size))

    def _mcmc_step(self, actions, cond, create_graph=False):
        """
        Performs a single Langevin dynamics step: 
        a_{t+1} = a_t - alpha * dE/da + noise
        """
        # Enable gradient calculation w.r.t actions (required for energy gradient)
        actions = actions.detach().requires_grad_(True)
        
        # Forward pass through EBT to get Energy scalar
        energy = self.model(actions, cond)
        energy_sum = energy.sum()
        
        # Calculate Gradient of Energy w.r.t Actions
        # We sum energy to get a scalar for autograd
        grad = torch.autograd.grad(energy_sum, actions, create_graph=create_graph, only_inputs=True, allow_unused=False)[0]
        
        # Clamp gradients for numerical stability
        grad = torch.clamp(grad, -1.0, 1.0)
        
        # Sample Langevin Noise
        noise = torch.randn_like(actions) * self.mcmc_noise_std
        
        # Gradient Descent (Minimize Energy) + Noise injection
        actions_new = actions - self.alpha * grad + noise
        
        return actions_new

    def forward(self, cond_features, gt_actions, state=None):
        """
        Training Forward: Unrolled Differentiable Optimization
        Args:
            cond_features: [B, Seq_Len, Dim] - Multimodal features from VLM
            gt_actions: [B, Horizon, Action_Dim] - Ground Truth Actions
            state: Optional robot state (not used in this simple EBT implementation)
        Returns:
            loss: Scalar MSE loss between refined actions and Ground Truth
        """
        if isinstance(cond_features, torch.Tensor):
            cond_features = [cond_features] * self.n_layer
        # 1. Initialize actions from Gaussian Noise
        current_actions = torch.randn_like(gt_actions)
        
        # 2. Run MCMC chain (Unrolled Loop)
        # We create the computation graph only on the last step to save memory,
        # but this allows backprop through the optimization trajectory if create_graph=True everywhere.
        # Typically for "Recurrent Back-Propagation" or "Deep Equilibrium" styles, specific handling is needed.
        # Here we use a simple unrolled approximation.
        for i in range(self.train_mcmc_steps):
            is_last_step = (i == self.train_mcmc_steps - 1)
            current_actions = self._mcmc_step(
                current_actions, 
                cond_features, 
                create_graph=is_last_step 
            )
            
        # 3. Compute Loss: Force the energy minimum to be at the GT location
        loss = F.mse_loss(current_actions, gt_actions)
        return loss

    # @torch.inference_mode()
    def predict_action(self, cond_features, state=None):
        """
        Inference: Iterative Energy Minimization
        """
        # Ensure input is a list for Layerwise Transformer
        if isinstance(cond_features, torch.Tensor):
            cond_features = [cond_features] * self.n_layer
            
        B = cond_features[0].shape[0]
        device = cond_features[0].device
        
        # 1. Initialize actions from Gaussian Noise
        # We perform initialization OUTSIDE the gradient loop to treat it as a constant starting point
        current_actions = torch.randn(
            B, self.horizon, self.action_dim, device=device
        )
        
        # 2. Iterative Refinement (Langevin Dynamics)
        # We explicitly enable gradients because inference usually runs in no_grad
        with torch.enable_grad():
            for _ in range(self.inf_mcmc_steps):
                # Detach current actions from previous iteration's graph to save memory
                # and prevent growing graph history. We treat the previous step as a constant "input".
                current_actions = current_actions.detach()
                current_actions.requires_grad_(True)
                
                # Perform one step. create_graph=False is fine here because we don't need
                # second-order derivatives (we just update the action value).
                current_actions = self._mcmc_step(
                    current_actions, 
                    cond_features, 
                    create_graph=False
                )
                
        # Detach final result to return a clean tensor
        return current_actions.detach()[:, -8:, :]

def get_action_model(config=None):
    """
    Factory method to initialize EBT_ActionHead from global config.
    Match signature with other action headers.
    """
    return EBT_ActionHead(config=config)