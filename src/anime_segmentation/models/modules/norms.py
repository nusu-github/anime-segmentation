"""Normalization helpers for BiRefNet modules."""

import inspect
import warnings
from typing import Any

from timm.layers import GroupNormAct
from torch import nn


def _filter_act_kwargs(act_layer: type[nn.Module], act_kwargs: dict | None) -> dict | None:
    if act_kwargs is None:
        return None
    if not isinstance(act_kwargs, dict):
        msg = f"act_kwargs must be a dict when using {act_layer.__name__}, got {type(act_kwargs).__name__}"
        raise TypeError(msg)

    try:
        signature = inspect.signature(act_layer)
        valid_keys = {name for name in signature.parameters if name != "self"}
    except (TypeError, ValueError):
        return act_kwargs

    filtered = {k: v for k, v in act_kwargs.items() if k in valid_keys}
    removed = sorted(set(act_kwargs) - set(filtered))
    if removed:
        warnings.warn(
            f"Ignoring unsupported act_kwargs for {act_layer.__name__}: {', '.join(removed)}",
            stacklevel=2,
        )
    return filtered


def _pick_group_count(
    num_channels: int,
    *,
    max_groups: int = 32,
    target_group_size: int = 8,
    min_group_size: int = 4,
    max_group_size: int = 16,
) -> int:
    """Pick a GroupNorm group count based on channel size.

    Prefers group sizes close to the target within [min_group_size, max_group_size],
    capped by max_groups. Falls back to the largest valid divisor.
    """
    max_groups = min(max_groups, num_channels)
    best_group = None
    best_score = None

    for groups in range(1, max_groups + 1):
        if num_channels % groups != 0:
            continue
        group_size = num_channels // groups
        if min_group_size <= group_size <= max_group_size:
            score = (abs(group_size - target_group_size), -groups)
            if best_score is None or score < best_score:
                best_score = score
                best_group = groups

    if best_group is not None:
        return best_group

    for groups in range(max_groups, 0, -1):
        if num_channels % groups == 0:
            return groups

    return 1


def group_norm(num_channels: int) -> nn.GroupNorm:
    """GroupNorm factory tuned for segmentation channel sizes."""
    target = 4 if num_channels < 32 else 8
    groups = _pick_group_count(
        num_channels,
        target_group_size=target,
    )
    return nn.GroupNorm(groups, num_channels)


def adaptive_group_norm_act(num_channels: int, **kwargs: Any) -> GroupNormAct:
    """GroupNormAct factory with adaptive group count based on channel size.

    This function is compatible with timm's ConvNormAct, which expects
    norm layers to accept apply_act, act_layer, act_kwargs, etc.
    """
    target = 4 if num_channels < 32 else 8
    num_groups = _pick_group_count(num_channels, target_group_size=target)
    if "act_layer" not in kwargs:
        act_kwargs = kwargs.get("act_kwargs")
        if isinstance(act_kwargs, dict) and "approximate" in act_kwargs:
            kwargs["act_layer"] = nn.GELU

    act_layer = kwargs.get("act_layer", nn.ReLU)
    act_kwargs = kwargs.get("act_kwargs")
    kwargs["act_kwargs"] = _filter_act_kwargs(act_layer, act_kwargs)
    return GroupNormAct(num_channels, num_groups=num_groups, **kwargs)
