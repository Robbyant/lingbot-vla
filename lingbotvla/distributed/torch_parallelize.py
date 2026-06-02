# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import types
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state_if_fully_sharded_module
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import create_selective_checkpoint_contexts, noop_context_fn

from ..models import load_model_weights
from ..utils import logging
from ..utils.import_utils import is_torch_version_greater_than
from .checkpoint import CheckpointFunction
from .fsdp import (
    clip_grad_norm_,
    init_fsdp_fn,
    parallel_init_fsdp_fn,
    parallel_load_safetensors,
    register_checkpoint_extension,
)
from .parallel_state import get_parallel_state
from .utils import get_module_from_path, set_module_from_path


if is_torch_version_greater_than("2.4"):
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor.parallel import parallelize_module


logger = logging.get_logger(__name__)


def _format_attr_path(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _resolve(root: Any, paths: List[tuple[str, ...]]) -> tuple[Optional[Any], Optional[tuple[str, ...]]]:
    for path in paths:
        current = root
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return current, path
    return None, None


def _resolve_required(root: Any, name: str, paths: List[tuple[str, ...]]) -> tuple[Any, tuple[str, ...]]:
    module, path = _resolve(root, paths)
    if module is None or path is None:
        candidates = ", ".join(_format_attr_path(candidate) for candidate in paths)
        raise RuntimeError(f"Could not locate {name}. Tried: {candidates}")
    return module, path


def _iter_unique_named_parameters(module: "nn.Module") -> List[tuple[str, torch.nn.Parameter]]:
    seen = set()
    params = []
    for name, param in module.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        params.append((name, param))
    return params


def _log_fsdp2_root_unit_summary(
    model: "nn.Module",
    sharded_modules: List[tuple[str, "nn.Module"]],
    shard_dtype: torch.dtype,
    world_size: int,
    topk: int = 12,
) -> None:
    if world_size <= 0:
        world_size = 1

    root_named_params = []
    sharded_param_ids = set()
    for _, module in sharded_modules:
        for _, param in _iter_unique_named_parameters(module):
            sharded_param_ids.add(id(param))

    for name, param in _iter_unique_named_parameters(model):
        if id(param) not in sharded_param_ids:
            root_named_params.append((name, param))

    root_param_count = sum(param.numel() for _, param in root_named_params)
    bytes_per_elem = torch.tensor([], dtype=shard_dtype).element_size()
    root_payload_bytes = root_param_count * bytes_per_elem
    ring_wire_bytes = root_payload_bytes * max(world_size - 1, 0) / max(world_size, 1)

    logger.info_rank0(
        "FSDP2 root unit summary: "
        f"params={root_param_count / 1e6:.2f}M, "
        f"payload={root_payload_bytes / 1024**3:.2f} GiB @ {shard_dtype}, "
        f"ring_wire_per_collective={ring_wire_bytes / 1024**3:.2f} GiB (world_size={world_size})."
    )
    logger.info_rank0(
        "FSDP2 comm estimate: "
        f"HEAD_AG≈{ring_wire_bytes / 1024**3:.2f} GiB wire, "
        f"Tail_RS≈{ring_wire_bytes / 1024**3:.2f} GiB wire."
    )

    sharded_module_summary = ", ".join(
        f"{name}({sum(param.numel() for _, param in _iter_unique_named_parameters(module)) / 1e6:.1f}M)"
        for name, module in sharded_modules
    )
    if sharded_module_summary:
        logger.info_rank0(f"FSDP2 explicitly sharded modules: {sharded_module_summary}")

    if root_named_params:
        root_named_params.sort(key=lambda item: item[1].numel(), reverse=True)
        formatted = ", ".join(
            f"{name}({param.numel() / 1e6:.1f}M)" for name, param in root_named_params[:topk]
        )
        logger.info_rank0(f"FSDP2 top root params: {formatted}")
    else:
        logger.info_rank0("FSDP2 top root params: <none>")


def verbose_fsdp_grouping(model, prefix="", depth=0):
    indent = "    " * depth

    for name, child in model.named_children():
        if isinstance(child, FullyShardedDataParallel):
            module_names = [m_name for m_name, _ in child.named_modules()][1:]  # [1:] 排除自身
            strategy = child.sharding_strategy
            logger.debug_rank0(f"{indent}├── [FSDP Group] {prefix}{name}")
            logger.debug_rank0(
                f"{indent}│   ├── Sharding Strategy: {strategy}, Mixed Precision: {child.mixed_precision}"
            )
            logger.debug_rank0(f"{indent}│   └── Contains Modules: {module_names}")

            verbose_fsdp_grouping(child, prefix=f"{prefix}{name}.", depth=depth + 1)
        else:
            verbose_fsdp_grouping(child, prefix=f"{prefix}{name}.", depth=depth)


def build_parallelize_model(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    sharding_plan: Optional[Dict[str, Any]] = None,
    enable_full_shard: bool = True,
    enable_mixed_precision: bool = True,
    enable_fp32: bool = False,
    enable_gradient_checkpointing: bool = True,
    basic_modules: Optional[List[str]] = None,
    fsdp_llm_blocks: bool = True,
    **kwargs,
) -> "nn.Module":
    """
    Applies parallel strategies to the model.
    """
    parallel_state = get_parallel_state()
    fsdp_no_shard_states = None

    if not parallel_state.fsdp_enabled:
        if kwargs.get("init_device") != "cuda":
            raise ValueError("Only FSDP training supports `init_device=cpu` or `init_device=meta`.")
        if kwargs.pop("enable_fsdp_offload", False):
            raise ValueError("Only FSDP training supports `enable_fsdp_offload`.")

    if enable_mixed_precision:  # upcast to float32 before feed it to optimizer
        model = model.float()

    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        logger.info_rank0("Enable gradient checkpointing.")
        use_reentrant = kwargs.pop("enable_reentrant", False)
        if use_reentrant:
            torch.utils.checkpoint.CheckpointFunction = CheckpointFunction

        ops_to_save = kwargs.pop("ops_to_save", None)
        context_fn = (
            partial(create_selective_checkpoint_contexts, ops_to_save) if ops_to_save is not None else noop_context_fn
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant, "context_fn": context_fn}
        )

    if parallel_state.tp_enabled:
        logger.info_rank0("Apply tensor parallel to the model.")
        model = parallelize_module(
            model,
            device_mesh=parallel_state.tp_mesh,
        )

    if parallel_state.ep_enabled:
        parallel_plan = model.get_parallel_plan()
        ep_param_suffix = parallel_plan.ep_param_suffix

        fqn2spec_info = parallel_plan.apply(model, parallel_state.ep_fsdp_device_mesh)
        fsdp_no_shard_states_fqn_to_module = parallel_plan.get_fsdp_no_shard_info(model)

        fsdp_no_shard_states = list(fsdp_no_shard_states_fqn_to_module.values())
        fsdp_no_shard_states_fqn = list(fsdp_no_shard_states_fqn_to_module.keys())
        logger.info_rank0(f"Apply expert parallel to the model successfully.\nEP modules: {fsdp_no_shard_states_fqn}.")
    else:
        fqn2spec_info = None
        ep_param_suffix = None
        fsdp_no_shard_states = None
        fsdp_no_shard_states_fqn = None

    if parallel_state.fsdp_enabled:
        logger.info_rank0(f"Apply data parallel to the model: {parallel_state.dp_mode}.")
        if parallel_state.dp_mode == "fsdp2":
            fsdp_kwargs = {
                "mesh": parallel_state.fsdp_mesh,
                "reshard_after_forward": enable_full_shard,
                **kwargs.pop("fsdp_kwargs", {}),
            }
            if enable_mixed_precision and not enable_fp32:
                logger.info_rank0("Enable mixed precision training.")
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.bfloat16,
                )
                fsdp_kwargs["mp_policy"] = mp_policy
            elif enable_fp32:
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.float32,
                )
                fsdp_kwargs["mp_policy"] = mp_policy
            shard_param_dtype = torch.float32 if enable_fp32 else torch.bfloat16
            explicitly_sharded_modules: List[tuple[str, nn.Module]] = []

            paligemma_with_expert, _ = _resolve(
                model,
                [
                    ("model", "paligemma_with_expert"),
                    ("paligemma_with_expert",),
                ],
            )

            qwenvl_with_expert, _ = _resolve(
                model,
                [
                    ("model", "qwenvl_with_expert"),
                    ("qwenvl_with_expert",),
                ],
            )

            llm_layers = None
            llm_layers_path = None
            expert_layers = None
            expert_layers_path = None
            if paligemma_with_expert is not None:
                llm_layers, llm_layers_path = _resolve_required(
                    paligemma_with_expert,
                    "Pi0 decoder layers",
                    [
                        ("paligemma", "model", "layers"),
                        ("paligemma", "language_model", "model", "layers"),
                        ("paligemma", "model", "language_model", "layers"),
                    ],
                )
                expert_layers, expert_layers_path = _resolve_required(
                    paligemma_with_expert,
                    "Gemma expert decoder layers",
                    [
                        ("gemma_expert", "model", "layers"),
                    ],
                )
            elif qwenvl_with_expert is not None:
                llm_layers, llm_layers_path = _resolve_required(
                    qwenvl_with_expert,
                    "Qwen decoder layers",
                    [
                        ("qwenvl", "model", "layers"),
                        ("qwenvl", "language_model", "model", "layers"),
                        ("qwenvl", "model", "language_model", "layers"),
                    ],
                )
                expert_layers, expert_layers_path = _resolve_required(
                    qwenvl_with_expert,
                    "qwen expert decoder layers",
                    [
                        ("qwen_expert", "model", "layers"),
                    ],
                )

            mp_fsdp_kwargs = {
                "mesh": parallel_state.fsdp_mesh,
                "reshard_after_forward": enable_full_shard,
                **kwargs.pop("fsdp_kwargs", {}),
            }

            mp_fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.bfloat16,
                )
            ignore_modules_in_mixed_precision = tuple()
            if hasattr(model, "get_ignore_modules_in_mixed_precision"):
                ignore_modules_in_mixed_precision = model.get_ignore_modules_in_mixed_precision()

            def apply_fsdp_to_decoder_blocks(module: "nn.Module") -> None:
                if module.__class__.__name__ in basic_modules or module.__class__ in ignore_modules_in_mixed_precision:
                    logger.debug(f"Apply FSDP2 to {module.__class__.__name__}.")
                    if module.__class__ in ignore_modules_in_mixed_precision:
                        fully_shard(module, **{k: v for k, v in fsdp_kwargs.items() if k != "mp_policy"})
                    else:
                        fully_shard(module, **fsdp_kwargs)

            if basic_modules:
                model.apply(apply_fsdp_to_decoder_blocks)
            elif fsdp_llm_blocks:
                if llm_layers is None or expert_layers is None:
                    raise RuntimeError(
                        "fsdp_llm_blocks=True requires a resolvable paligemma_with_expert module. "
                        "Expected one of model.paligemma_with_expert or model.model.paligemma_with_expert."
                    )
                if not hasattr(llm_layers, "__iter__") or not hasattr(expert_layers, "__iter__"):
                    raise TypeError("Expected 'layers' to be a module list or container.")

                logger.info_rank0(
                    "Applying FSDP to "
                    f"{len(llm_layers)} Pi0/Qwen layers via {_format_attr_path(llm_layers_path)} "
                    f"and {len(expert_layers)} Gemma expert layers via {_format_attr_path(expert_layers_path)}."
                )
                for i, layer in enumerate(llm_layers):
                    logger.debug(f"Sharding layer {i} ({layer.__class__.__name__})")
                    fully_shard(layer, **fsdp_kwargs)
                    explicitly_sharded_modules.append((f"{_format_attr_path(llm_layers_path)}[{i}]", layer))
                for i, layer in enumerate(expert_layers):
                    logger.debug(f"Sharding layer {i} ({layer.__class__.__name__})")
                    fully_shard(layer, **fsdp_kwargs)
                    explicitly_sharded_modules.append((f"{_format_attr_path(expert_layers_path)}[{i}]", layer))

                extra_shard_targets = [
                    (
                        "Pi0/Qwen embed_tokens",
                        [
                            ("paligemma", "model", "embed_tokens"),
                            ("paligemma", "language_model", "model", "embed_tokens"),
                            ("paligemma", "model", "language_model", "embed_tokens"),
                            ("qwenvl", "model", "embed_tokens"),
                            ("qwenvl", "language_model", "model", "embed_tokens"),
                            ("qwenvl", "model", "language_model", "embed_tokens"),
                        ],
                    ),
                    (
                        "Pi0/Qwen lm_head",
                        [
                            ("paligemma", "lm_head"),
                            ("paligemma", "language_model", "lm_head"),
                            ("qwenvl", "lm_head"),
                            ("qwenvl", "language_model", "lm_head"),
                        ],
                    ),
                    (
                        "Pi0/Qwen visual.patch_embed",
                        [
                            ("paligemma", "visual", "patch_embed"),
                            ("paligemma", "model", "visual", "patch_embed"),
                            ("qwenvl", "visual", "patch_embed"),
                            ("qwenvl", "model", "visual", "patch_embed"),
                        ],
                    ),
                    (
                        "Pi0/Qwen visual.merger",
                        [
                            ("paligemma", "visual", "merger"),
                            ("paligemma", "model", "visual", "merger"),
                            ("qwenvl", "visual", "merger"),
                            ("qwenvl", "model", "visual", "merger"),
                        ],
                    ),
                    (
                        "expert_visual",
                        [
                            ("expert_visual",),
                        ],
                    ),
                    (
                        "expert_visual_mlp",
                        [
                            ("expert_visual_mlp",),
                        ],
                    ),
                ]
                for target_name, candidate_paths in extra_shard_targets:
                    target_module, target_path = _resolve(paligemma_with_expert, candidate_paths)
                    if target_module is None or target_path is None:
                        continue
                    logger.debug(f"Sharding {target_name} via {_format_attr_path(target_path)}")
                    fully_shard(target_module, **fsdp_kwargs)
                    explicitly_sharded_modules.append((_format_attr_path(target_path), target_module))

                _log_fsdp2_root_unit_summary(
                    model=model,
                    sharded_modules=explicitly_sharded_modules,
                    shard_dtype=shard_param_dtype,
                    world_size=parallel_state.fsdp_mesh.size(),
                )
            llm_layers, llm_path = _resolve(model.model.qwenvl_with_expert.qwenvl, [
                ("model", "layers"),                      # transformers 4.5x
                ("language_model", "model", "layers"),    # 旧版本兼容
            ])
            
            if llm_layers is None or not hasattr(llm_layers, "__iter__"):
                raise RuntimeError(
                    "Could not locate Qwen2.5-VL decoder layers. ... "
                    "sharding would silently fall back and produce a 5+ GB root AllGather."
                )
            for layer in llm_layers:
                if layer.__class__.__name__ == "Qwen2_5_VLDecoderLayer" or layer.__class__.__name__ == "Qwen2_5_VLVisionBlock":
                    logger.info_rank0(f"Apply FSDP2 to {layer.__class__.__name__}.")
                    fully_shard(layer, **mp_fsdp_kwargs)
            fully_shard(model, **mp_fsdp_kwargs)

            if kwargs.get("init_device") == "meta":
                if weights_path is None:
                    # shard init empty model with fsdp2
                    model.to_empty(device="cuda")
                    model.init_weights()
                else:
                    from torch.distributed.tensor import distribute_tensor

                    load_model_weights(model, weights_path, "cuda", dtensor_factory=distribute_tensor)
            llm_layers, llm_path = _resolve(model.model.qwenvl_with_expert.qwenvl, [
                ("model", "layers"),                      # transformers 4.5x
                ("language_model", "model", "layers"),    # 旧版本兼容
            ])
            
            if llm_layers is None or not hasattr(llm_layers, "__iter__"):
                raise RuntimeError(
                    "Could not locate Qwen2.5-VL decoder layers. ... "
                    "sharding would silently fall back and produce a 5+ GB root AllGather."
                )
            for layer in llm_layers:
                if layer.__class__.__name__ == "Qwen2_5_VLDecoderLayer" or layer.__class__.__name__ == "Qwen2_5_VLVisionBlock":
                    logger.info_rank0(f"Apply FSDP2 to {layer.__class__.__name__}.")
                    fully_shard(layer, **mp_fsdp_kwargs)

            # Eliminate backward all-gather by disabling reshard_after_backward.
            if not enable_full_shard and hasattr(model, "set_reshard_after_backward"):
                model.set_reshard_after_backward(False, recurse=True)
                logger.info_rank0("FSDP2: set_reshard_after_backward(False) applied to eliminate backward all-gather.")

        elif parallel_state.dp_mode == "fsdp1":
            wrap_policy = partial(
                lambda_auto_wrap_policy, lambda_fn=lambda module: module.__class__.__name__ in basic_modules
            )

            # set fsdp/hsdp sharding strategy
            if parallel_state.fsdp_mesh.ndim > 1 and parallel_state.fsdp_mesh.size() > 1:
                strategy = ShardingStrategy.HYBRID_SHARD
            else:
                strategy = ShardingStrategy.FULL_SHARD

            fsdp_kwargs = {
                "auto_wrap_policy": wrap_policy,
                "ignored_states": fsdp_no_shard_states,
                "device_id": torch.cuda.current_device(),
                "sharding_strategy": strategy if enable_full_shard else ShardingStrategy.NO_SHARD,
                "use_orig_params": True,
            }

            fsdp_kwargs["device_mesh"] = parallel_state.fsdp_mesh

            fsdp_kwargs.update(kwargs.pop("fsdp_kwargs", {}))

            if enable_mixed_precision:
                logger.info_rank0("Enable mixed precision training.")
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                )
                if hasattr(model, "get_ignore_modules_in_mixed_precision"):
                    mixed_precision._module_classes_to_ignore += model.get_ignore_modules_in_mixed_precision()

                fsdp_kwargs["mixed_precision"] = mixed_precision

            if kwargs.get("init_device") == "cpu":
                logger.info_rank0("Enable rank0-only initialization.")
                fsdp_kwargs["sync_module_states"] = True
                if parallel_state.global_rank != 0:
                    fsdp_kwargs["param_init_fn"] = init_fsdp_fn(model, device="cuda")
            elif kwargs.get("init_device") == "meta":
                # assert weights_path is not None, "`weights_path` must be provided when `init_device=meta` for fsdp1."

                logger.info_rank0("Enable meta initialization.")
                if weights_path is None:
                    logger.info_rank0("weights_path is None during meta initialization.")

                ignore_param_names = (
                    [".".join([fqn, k]) for fqn in fsdp_no_shard_states_fqn for k in ep_param_suffix]
                    if fsdp_no_shard_states_fqn is not None
                    else None
                )
                shard_states = (
                    parallel_load_safetensors(weights_path, ignore_param_name=ignore_param_names)
                    if weights_path
                    else kwargs.get("state_dict", {})
                )
                fsdp_kwargs["param_init_fn"] = parallel_init_fsdp_fn(
                    model, shard_states, ignore_param_name=ignore_param_names
                )

            if kwargs.pop("enable_fsdp_offload", False):
                logger.info_rank0("Enable offloading for parameters & gradients & optimizer states.")
                fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

            if kwargs.pop("enable_forward_prefetch", False):
                fsdp_kwargs["forward_prefetch"] = True
            else:
                fsdp_kwargs["forward_prefetch"] = False
                fsdp_kwargs["backward_prefetch"] = None

            # FULLY_SHARD first
            model = FullyShardedDataParallel(model, **fsdp_kwargs)

            if fsdp_no_shard_states is not None:
                # apply NO_SHARD the ignored_states, but wrap into DDP
                if parallel_state.ep_fsdp_mesh["ep_fsdp"].size() == 1:
                    moe_sharding_strategy = ShardingStrategy.NO_SHARD
                    ep_fsdp_device_mesh = parallel_state.fsdp_mesh
                else:
                    moe_sharding_strategy = ShardingStrategy.FULL_SHARD
                    ep_fsdp_device_mesh = parallel_state.ep_fsdp_mesh["ep_fsdp"]

                logger.info_rank0(f"Apply {moe_sharding_strategy} states on '{fsdp_no_shard_states_fqn}'.")
                fsdp_kwargs.pop("ignored_states", None)
                fsdp_kwargs.pop("auto_wrap_policy", None)
                fsdp_kwargs["sharding_strategy"] = moe_sharding_strategy
                fsdp_kwargs["device_mesh"] = ep_fsdp_device_mesh
                logger.info_rank0(f"{ep_fsdp_device_mesh=}")
                for fqn in fsdp_no_shard_states_fqn:
                    no_shard_module = get_module_from_path(model, fqn)
                    if kwargs.get("init_device") == "meta":
                        specific_param_name = [".".join([fqn, k]) for k in ep_param_suffix]
                        shard_states = (
                            parallel_load_safetensors(weights_path, specific_param_name=specific_param_name)
                            if weights_path
                            else {}
                        )
                        if weights_path:
                            for suffix in ep_param_suffix:
                                shard_states[suffix] = shard_states.pop(".".join([fqn, suffix]))
                        fsdp_kwargs["param_init_fn"] = parallel_init_fsdp_fn(
                            no_shard_module, shard_states, specific_param_name=ep_param_suffix
                        )
                    fsdp_module = FullyShardedDataParallel(no_shard_module, **fsdp_kwargs)
                    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(fsdp_module)
                    fsdp_state._gradient_postdivide_factor *= parallel_state.ep_size
                    set_module_from_path(model, fqn, fsdp_module)

            _lazy_init(model, model)

            # Apply fsdp extension to FSDP model
            save_hook_mesh = parallel_state.ep_fsdp_device_mesh if parallel_state.ep_enabled else None
            logger.info_rank0("Register Checkpoints Extension hook to the model")
            register_checkpoint_extension(
                fsdp_model=model,
                save_hook_mesh=save_hook_mesh,
                fqn2spec_info=fqn2spec_info,
            )

            if parallel_state.ep_enabled:
                model.clip_grad_norm_ = types.MethodType(clip_grad_norm_, model)

            verbose_fsdp_grouping(model)
        else:
            ddp_kwargs = {"device_ids": [parallel_state.local_rank]}
            if enable_mixed_precision:
                logger.info_rank0("Enable mixed precision training.")
                if enable_fp32:
                    mixed_precision = MixedPrecision(
                        param_dtype=torch.float32,
                        reduce_dtype=torch.float32,
                        buffer_dtype=torch.float32,
                    )
                else:
                    mixed_precision = MixedPrecision(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        buffer_dtype=torch.bfloat16,
                    )
                ddp_kwargs["mixed_precision"] = mixed_precision

            model = DDP(model, **ddp_kwargs)

    return model
