"""Normalization helpers for BiRefNet modules."""

from torch import nn


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
