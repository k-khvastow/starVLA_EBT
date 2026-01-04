"""
Qwen-PI-EBT Framework
A lightweight implementation that Qwen2.5-vl + Energy-Based Transformer (EBT) head 
to directly predict continuous actions via differentiable Langevin Dynamics.
"""
from typing import List, Optional, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.EBT_ActionHeader import EBT_ActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

####################################################
# ⚠️ Warning: This framework has been modified to use EBT.
####################################################

@FRAMEWORK_REGISTRY.register("QwenPI")
class Qwen_PI(baseframework):
    """
    Multimodal vision-language-action model (EBT Version).

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Energy-Based Transformer (EBT) head for implicit action modeling
      
    Focus: Predict future continuous actions conditioned on images + instruction
           using iterative energy minimization (Langevin Dynamics).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # dynamic get llm config
        num_vl_layers, llm_hidden_size = 36, self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # Initialize EBT Action Head
        # Ensure config.framework.action_model parameters are set for EBT (e.g. inference_steps)
        self.action_model = EBT_ActionHead(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): MSE loss from unrolled Langevin dynamics.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # Use Last Hidden State for EBT
            # Unlike LayerwiseFM, standard EBT usually takes the top-level embedding sequence.
            # You can also use a specific layer index from config if desired.
            last_hidden = qwenvl_outputs.hidden_states[-1] 

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 标签对齐：取最后 chunk_len 段
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )

            # The EBT_ActionHead handles the MCMC unrolling internally based on 'train_mcmc_steps'.
            # We do NOT need to manually repeat the hidden states here unless doing parallel chains 
            # (which increases memory usage significantly).
            action_loss = self.action_model(last_hidden, actions_target, state_tensor)
        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action( # TODO align  predict_action with forward, make api more flexible
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：Iterative Energy Minimization (Langevin Dynamics).

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL
          3. Generate initial noise
          4. Refine actions by descending the energy landscape (MCMC)

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim]
        """
        from deployment.model_server.tools.image_tools import to_pil_preserve
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        state_tensor = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        
        # Step 4: Action Expert Forward (Inference MCMC)
        with torch.autocast("cuda", dtype=torch.float32):
            # predict_action in EBT_ActionHead performs the Langevin loop
            pred_actions = self.action_model.predict_action(last_hidden, state_tensor)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    
    if not hasattr(cfg.framework.action_model, "inference_steps"):
        cfg.framework.action_model.inference_steps = 10
    if not hasattr(cfg.framework.action_model, "repeated_diffusion_steps"):
        cfg.trainer.repeated_diffusion_steps = 4 # repurposed for train MCMC steps

    model = Qwen_PI(cfg)
    print(model)

    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake instruction for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")