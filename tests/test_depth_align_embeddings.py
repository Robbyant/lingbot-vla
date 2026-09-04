"""Regression tests for query-mode depth alignment parameters."""

import torch
from torch import nn

from lingbotvla.models.vla.pi0.modeling_lingbot_vla import FlowMatching


_DEPTH_CONFIG = {
    "llm": {"image_token_size": 2, "image_input_size": 4, "dim_out": 8},
    "depth": {
        "token_size": 2,
        "input_size": 4,
        "model_type": "MoRGBD",
        "num_backbone_tokens": 4,
        "dim_head": 2,
        "dim_out": 8,
        "num_layers": 1,
        "num_heads": 1,
        "ff_mult": 2,
    },
    "mode": "query",
    "num_task_tokens": 2,
}


def _depth_alignment_module() -> FlowMatching:
    """Build only the depth heads so the test does not allocate a VLA backbone."""
    model = object.__new__(FlowMatching)
    nn.Module.__init__(model)
    model.init_depth_heads(_DEPTH_CONFIG)
    return model


def test_depth_alignment_embeddings_are_trainable_parameters():
    torch.manual_seed(0)
    model = _depth_alignment_module()
    parameter = model.depth_align_embs

    assert isinstance(parameter, nn.Parameter)
    assert parameter.dtype == torch.bfloat16
    assert parameter.requires_grad
    assert callable(parameter.requires_grad_)
    assert dict(model.named_parameters())["depth_align_embs"] is parameter

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = parameter.detach().clone()
    parameter.float().square().mean().backward()
    optimizer.step()

    assert not torch.equal(before, parameter.detach())


def test_depth_alignment_embeddings_round_trip_in_state_dict():
    model = _depth_alignment_module()
    state = model.state_dict()

    assert "depth_align_embs" in state

    restored = _depth_alignment_module()
    result = restored.load_state_dict(state, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert torch.equal(restored.depth_align_embs, state["depth_align_embs"])
