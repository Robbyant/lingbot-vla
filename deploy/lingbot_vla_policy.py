"""Utils for evaluating the OpenVLA policy."""

import json
import argparse
import os
import time
import random
from collections import deque
import torchvision
import yaml
from types import SimpleNamespace

from glob import glob
from tqdm import tqdm
from safetensors import safe_open
from pathlib import Path

import transformers
from transformers import (
    AutoConfig,
)
from typing import Union
import numpy as np

import torch
from PIL import Image
import torch.nn.functional as F
from torch import Tensor, nn
from packaging.version import Version

from lerobot.configs.policies import PreTrainedConfig
from .websocket_policy_server import WebsocketPolicyServer
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
from lingbotvla.data.vla_data.utils import FeatureTransform
from lingbotvla.models import build_processor

def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
set_seed_everywhere(42)

class PolicyPreprocessMixin:

    @torch.no_grad
    def select_action(
        self, observation: dict[str, Tensor], use_bf16: bool = False, noise: Tensor | None = None, num_denoising_step : int = 10
    ):
        self.eval()
        device = next(self.parameters()).device.type
        if use_bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        s1 = time.time()
        
        if len(observation['images'].shape) == 4:
            observation['images'] = observation['images'].unsqueeze(0)
            observation['img_masks'] = observation['img_masks'].unsqueeze(0)

        actions = self.model.sample_actions(
            observation['images'].to(dtype=dtype, device=device), 
            observation['img_masks'].to(device=device), 
            observation['lang_tokens'].unsqueeze(0).to(device=device), 
            observation['lang_masks'].unsqueeze(0).to(device=device), 
            observation['state'].unsqueeze(0).to(dtype=dtype, device=device), 
            num_steps = num_denoising_step
        )
        print('sample action time: ', time.time()-s1)
        
        observation['actions'] = actions.squeeze(0).to(dtype=torch.float32, device='cpu')
        if use_bf16:
            observation['state'] = observation['state'].to(dtype=torch.float32)
        data = self.feature_transform.unapply(observation)
        return data

class LingBotVlaInferencePolicy(PolicyPreprocessMixin, LingbotVlaPolicy):
    pass # Only combine necessary functions

def merge_qwen_config(policy_config, qwen_config):
    if hasattr(qwen_config, 'to_dict'):
        config_dict = qwen_config.to_dict()
    else:
        config_dict = qwen_config

    text_keys = {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
        "tokenizer_path",
    }

    for key in text_keys:
        if key in config_dict:
            setattr(policy_config, key, config_dict[key])
            print(f"✅ Merged LLM: {key} = {config_dict[key]}")

    if "vision_config" in config_dict:
        policy_config.vision_config = qwen_config.vision_config
    else:
        print("⚠️ Warning: 'vision_config' not found in qwen_config!")

    return policy_config


class LingbotVLAServer:
    '''
    policy wrapper to support action ensemble or chunk execution
    '''
    def __init__(
        self,
        path_to_pi_model="",
        adaptive_ensemble_alpha=0.1,
        action_ensemble_horizon=8,
        use_length=1, # to control the execution length of the action chunk, -1 denotes using action ensemble
        use_bf16=True,
        use_fp32=False,
        robot_norm_path: str = None,
        num_denoising_step=10,
        use_compile=False
    ) -> None:
        assert not (use_bf16 and use_fp32), 'Bfloat16 or Float32!!!'
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.use_length = use_length
        self.use_compile = use_compile
        self.num_denoising_step = num_denoising_step

        self.robot_norm_path = robot_norm_path
        
        self.vla = self.load_vla(path_to_pi_model)
        self.vla = self.vla.eval()
        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
        elif use_fp32:
            self.vla.model.float()
        self.global_step = 0
        self.last_action_chunk = None
        self.use_bf16 = use_bf16

    def load_vla(self, path_to_pi_model) -> LingbotVlaPolicy:
        # load model
    
        print(f"loading model from: {path_to_pi_model}")
        config = PreTrainedConfig.from_pretrained(path_to_pi_model)
        
        # load training config
        training_config_path = Path(path_to_pi_model)/'lingbotvla_cli.yaml'
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)
        f.close()

        # update model config according to training config
        config_kwargs = {**training_config['model'], **training_config['train']}
        missing_config_kwargs = {k: v for k, v in config_kwargs.items() if not hasattr(config, k)}
        config.__dict__.update(missing_config_kwargs)

        # Set attention_implementation to 'eager' to speed up evaluation.
        config.attention_implementation = 'eager'
        
        # set base model according to training config
        base_model_path = os.environ.get('QWEN25_PATH', 'Qwen/Qwen2.5-VL-3B-Instruct')
        config.tokenizer_path = base_model_path
        
        qwen_config = AutoConfig.from_pretrained(base_model_path)
        config = merge_qwen_config(config, qwen_config)

        # load processors
        if 'vocab_size' in training_config['model'] and training_config['model']['vocab_size'] != 0:
            config.vocab_size = training_config['model']['vocab_size']
        config.use_cache = True
        self.processor = build_processor(base_model_path)
        self.language_tokenizer = self.processor.tokenizer
        data_config = SimpleNamespace(**training_config['data'])
        data_config.max_state_dim = config.max_state_dim
        data_config.max_action_dim = config.max_action_dim
        data_config.resize_imgs_with_padding = config.resize_imgs_with_padding
        data_config.tokenizer_max_length = config.tokenizer_max_length
        
        print('Initializing model ... ')
        policy = LingBotVlaInferencePolicy(config, tokenizer_path=base_model_path)

        all_safetensors = glob(os.path.join(path_to_pi_model, "*.safetensors"))
        merged_weights = {}

        for file_path in tqdm(all_safetensors):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    merged_weights[key] = f.get_tensor(key)
        policy.load_state_dict(merged_weights, strict=True)
        # Device selection: preserve CUDA behavior, enable XPU, fall back to CPU.
        if torch.cuda.is_available():
            policy.to("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            policy.to("xpu")
        else:
            policy.to("cpu")
        
        if self.use_compile:
            policy.model.qwenvl_with_expert = torch.compile(policy.model.qwenvl_with_expert)

        if self.robot_norm_path is None:
            self.robot_norm_path = data_config.norm_stats_file

        policy.feature_transform = None
        self.data_config = data_config
        self.config = config

        print('Model initialized ... ')

        return policy

    def reset(self, robo_name) -> None:

        self.global_step = 0
        self.last_action_chunk = None

        image_processor = self.processor.image_processor
        robot_config = f'configs/robot_configs/{robo_name}.yaml'

        feature_transform = FeatureTransform(robot_config, self.data_config, \
                    self.language_tokenizer, image_processor, \
                    chunk_size=self.config.chunk_size,
                    norm_stats_path=self.robot_norm_path)
        # Load data processors
        self.vla.feature_transform = feature_transform

    def resize_image(self, observation):
        image_features  = self.vla.feature_transform.org_features['images']
        for image_feature in image_features:
            assert image_feature in observation
            assert len(observation[image_feature].shape)==3 and observation[image_feature].shape[-1] == 3
            image = observation[image_feature]
            img_pil = Image.fromarray(image)
            image_size = getattr(self.data_config, 'img_size', 224)
            img_pil = img_pil.resize((image_size, image_size), Image.BILINEAR)

            # img_resized shape: C*H*W
            img_resized = np.transpose(np.array(img_pil), (2,0,1))  # (3,224,224)
            observation[image_feature] = img_resized / 255.

    def infer(self, observation):
        """Generates an action with the VLA policy."""

        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if 'reset' in observation and observation['reset']:
            self.reset(robo_name=observation['robo_name'])
            return dict(action = None)
        self.resize_image(observation)
        for k, v in observation.items():
            if isinstance(v, np.ndarray):
                observation[k] = torch.from_numpy(v)
        
        for action_feature in self.vla.feature_transform.org_features['actions']:
            if action_feature not in observation:
                observation[action_feature] =  torch.zeros(self.vla.feature_transform.chunk_size, observation[self.vla.feature_transform.org_features['states'][0]].shape[0])

        observation[self.vla.feature_transform.org_features['actions'][0]+'_is_pad'] = torch.zeros(observation[self.vla.feature_transform.org_features['actions'][0]].shape[0])
        
        observation = self.vla.feature_transform.apply(observation)
        if self.use_bf16:
            observation['state'] = observation['state'].to(torch.bfloat16)
        output = self.vla.select_action(observation, self.use_bf16, num_denoising_step=self.num_denoising_step)
    
        action_chunk = {}
        for output_key in output.keys():
            if output_key in self.vla.feature_transform.org_features['actions']:
                assert self.use_length <= output[output_key].shape[0]
                action_length = self.use_length if self.use_length > 0 else output[output_key].shape[0]
                action_chunk[output_key] = output[output_key][ :action_length,:].float().cpu().numpy()
        self.global_step+=1
        return action_chunk

def main():
    parser = argparse.ArgumentParser(description="Start WebSocket policy server")

    parser.add_argument(
        "--model_path",
        type=str,
    )

    parser.add_argument(
        "--use_length",
        type=int,
        default=25,
        help="Usage length of the action chunk"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8006,
        help="WebSocket server port"
    )
    parser.add_argument('--norm_path',   type=str, default=None, help='norm file path of training data')
    parser.add_argument("--num_denoising_step", type=int, default=10, help="num of denoising step")
    parser.add_argument("--use_compile", action='store_true', help="use torch compile or not")

    args = parser.parse_args()

    model = LingbotVLAServer(args.model_path, use_length=args.use_length, robot_norm_path=args.norm_path, num_denoising_step=args.num_denoising_step, use_compile=args.use_compile)
    model_server = WebsocketPolicyServer(model, port=args.port)
    model_server.serve_forever()



if __name__ == "__main__":
    main()