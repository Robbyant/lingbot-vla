#!/usr/bin/env python3
"""Export LingBot-VLA assets for pi.cpp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from deploy.lingbot_vla_policy import LingbotVLAServer


SEED = 1234


class LingBotAction(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, denoise_steps: int) -> None:
        super().__init__()
        self.model = model
        self.denoise_steps = denoise_steps

    def forward(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            num_steps=self.denoise_steps,
        )


def _dataset(repo_or_path: str) -> LeRobotDataset:
    path = Path(repo_or_path)
    if path.is_absolute() and path.exists():
        return LeRobotDataset(path.name, root=path)
    return LeRobotDataset(repo_or_path)


def _image_to_hwc_uint8(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _prepare_observation(server: LingbotVLAServer, data_path: str) -> dict[str, torch.Tensor]:
    dataset = _dataset(data_path)
    observation = dict(dataset[0])
    for image_key in server.vla.feature_transform.org_features["images"]:
        observation[image_key] = _image_to_hwc_uint8(observation[image_key])
    server.resize_image(observation)

    for key, value in list(observation.items()):
        if isinstance(value, np.ndarray):
            observation[key] = torch.from_numpy(value)

    state_key = server.vla.feature_transform.org_features["states"][0]
    action_key = server.vla.feature_transform.org_features["actions"][0]
    if action_key not in observation:
        observation[action_key] = torch.zeros(
            server.vla.feature_transform.chunk_size,
            observation[state_key].shape[0],
        )
    observation[f"{action_key}_is_pad"] = torch.zeros(observation[action_key].shape[0])
    transformed = server.vla.feature_transform.apply(observation)

    images = transformed["images"]
    img_masks = transformed["img_masks"]
    if images.ndim == 4:
        images = images.unsqueeze(0)
        img_masks = img_masks.unsqueeze(0)
    return {
        "images": images.to(dtype=torch.bfloat16, device="cuda"),
        "img_masks": img_masks.to(device="cuda"),
        "lang_tokens": transformed["lang_tokens"].unsqueeze(0).to(device="cuda"),
        "lang_masks": transformed["lang_masks"].unsqueeze(0).to(device="cuda"),
        "state": transformed["state"].unsqueeze(0).to(dtype=torch.bfloat16, device="cuda"),
    }


def _array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LingBot-VLA assets for pi.cpp")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="./lingbot_picpp")
    parser.add_argument("--robo_name", default=None)
    parser.add_argument("--norm_path", default=None)
    parser.add_argument("--denoise_steps", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    output_dir = Path(args.output_dir)
    onnx_dir = output_dir / "onnx"
    io_dir = output_dir / "io"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    io_dir.mkdir(parents=True, exist_ok=True)

    server = LingbotVLAServer(
        path_to_pi_model=args.model_path,
        robot_norm_path=args.norm_path,
        use_length=50,
        use_bf16=True,
        num_denoising_step=args.denoise_steps,
        use_compile=False,
    )
    robo_name = args.robo_name if args.robo_name is not None else server.data_config.data_name
    server.reset(robo_name)
    tensors = _prepare_observation(server, args.data_path)
    noise = torch.randn(
        (
            tensors["state"].shape[0],
            server.config.n_action_steps,
            server.config.max_action_dim,
        ),
        dtype=torch.bfloat16,
        device="cuda",
    )
    wrapper = LingBotAction(server.vla.model, args.denoise_steps).eval()
    inputs = (
        tensors["images"],
        tensors["img_masks"],
        tensors["lang_tokens"],
        tensors["lang_masks"],
        tensors["state"],
        noise,
    )
    input_names = ["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"]
    with torch.inference_mode():
        actions = wrapper(*inputs)
        torch.onnx.export(
            wrapper,
            inputs,
            onnx_dir / "action.onnx",
            input_names=input_names,
            output_names=["actions"],
            opset_version=19,
            do_constant_folding=True,
            dynamo=False,
        )

    np.savez(
        io_dir / "action_inputs.npz",
        images=_array(tensors["images"]),
        img_masks=tensors["img_masks"].detach().cpu().numpy(),
        lang_tokens=tensors["lang_tokens"].detach().cpu().numpy(),
        lang_masks=tensors["lang_masks"].detach().cpu().numpy(),
        state=_array(tensors["state"]),
        noise=_array(noise),
    )
    np.save(io_dir / "actions.npy", _array(actions))
    manifest = {
        "model": "lingbot",
        "source_model": args.model_path,
        "robot": robo_name,
        "precision": "bf16",
        "seed": SEED,
        "onnx": {"action": "onnx/action.onnx"},
        "io": {"inputs": "io/action_inputs.npz", "actions": "io/actions.npy"},
        "inputs": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in zip(input_names, inputs, strict=True)
        },
        "outputs": {"actions": {"shape": list(actions.shape), "dtype": str(actions.dtype)}},
        "action": {
            "horizon": int(server.config.n_action_steps),
            "dim": int(server.config.max_action_dim),
            "denoise_steps": int(args.denoise_steps),
        },
        "features": {
            "images": list(server.vla.feature_transform.org_features["images"]),
            "states": list(server.vla.feature_transform.org_features["states"]),
            "actions": list(server.vla.feature_transform.org_features["actions"]),
        },
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
