"""Data validation and distribution monitoring for synthesis pipeline.

Provides validation assertions and distribution tracking to ensure
synthetic data quality and detect issues early.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation.

    Attributes:
        is_valid: Whether the data passed all validation checks.
        errors: List of error messages (validation failures).
        warnings: List of warning messages (non-critical issues).
        stats: Dictionary of computed statistics.

    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class DataValidator:
    """Validate synthetic data for quality assurance.

    Performs automatic assertions on image-mask pairs to catch
    issues early in the synthesis pipeline.
    """

    def __init__(
        self,
        require_binary: bool = True,
        require_shape_match: bool = True,
        min_fg_ratio: float = 0.0,
        max_fg_ratio: float = 0.95,
        value_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Initialize data validator.

        Args:
            require_binary: Require mask to be binary (0 or 1 only).
            require_shape_match: Require image and mask spatial dims to match.
            min_fg_ratio: Minimum foreground ratio (0 for negative examples).
            max_fg_ratio: Maximum foreground ratio.
            value_range: Expected (min, max) value range for image.

        """
        self.require_binary = require_binary
        self.require_shape_match = require_shape_match
        self.min_fg_ratio = min_fg_ratio
        self.max_fg_ratio = max_fg_ratio
        self.value_range = value_range

    def validate(self, image: Tensor, mask: Tensor) -> ValidationResult:
        """Validate an image-mask pair.

        Args:
            image: Image tensor [C, H, W] or [B, C, H, W].
            mask: Mask tensor [1, H, W] or [B, 1, H, W].

        Returns:
            ValidationResult with validation status and diagnostics.

        """
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}

        # Handle batch dimension
        if image.ndim == 4:
            image = image[0]
        if mask.ndim == 4:
            mask = mask[0]

        # Check shapes
        _, img_h, img_w = image.shape
        _, mask_h, mask_w = mask.shape

        stats["image_shape"] = (img_h, img_w)
        stats["mask_shape"] = (mask_h, mask_w)

        if self.require_shape_match and (img_h != mask_h or img_w != mask_w):
            errors.append(f"Shape mismatch: image ({img_h}, {img_w}) vs mask ({mask_h}, {mask_w})")

        # Check image value range
        img_min = float(image.min().item())
        img_max = float(image.max().item())
        stats["image_min"] = img_min
        stats["image_max"] = img_max

        if img_min < self.value_range[0] - 1e-6:
            errors.append(f"Image min ({img_min:.4f}) below expected ({self.value_range[0]})")
        if img_max > self.value_range[1] + 1e-6:
            errors.append(f"Image max ({img_max:.4f}) above expected ({self.value_range[1]})")

        # Check mask is binary
        mask_values = mask.unique()
        stats["mask_unique_values"] = len(mask_values)

        if self.require_binary:
            non_binary = ~((mask == 0) | (mask == 1))
            if non_binary.any():
                n_non_binary = non_binary.sum().item()
                errors.append(f"Mask has {n_non_binary} non-binary values")

        # Check foreground ratio
        fg_ratio = (mask > 0.5).float().mean().item()
        stats["fg_ratio"] = fg_ratio

        if fg_ratio < self.min_fg_ratio:
            warnings.append(f"FG ratio ({fg_ratio:.4f}) below min ({self.min_fg_ratio})")
        if fg_ratio > self.max_fg_ratio:
            errors.append(f"FG ratio ({fg_ratio:.4f}) above max ({self.max_fg_ratio})")

        # Check for NaN/Inf
        if torch.isnan(image).any():
            errors.append("Image contains NaN values")
        if torch.isinf(image).any():
            errors.append("Image contains Inf values")
        if torch.isnan(mask).any():
            errors.append("Mask contains NaN values")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )


class DistributionMonitor:
    """Monitor and log distribution statistics during synthesis.

    Tracks foreground ratio distribution, character count distribution,
    and other statistics for quality monitoring.
    """

    def __init__(
        self,
        log_dir: str | Path,
        bins: int = 50,
    ) -> None:
        """Initialize distribution monitor.

        Args:
            log_dir: Directory for saving distribution logs.
            bins: Number of histogram bins.

        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.bins = bins

        # Tracked distributions
        self._fg_ratios: list[float] = []
        self._k_counts: list[int] = []
        self._sample_count = 0

    def record(
        self,
        fg_ratio: float,
        k: int | None = None,
    ) -> None:
        """Record a sample's statistics.

        Args:
            fg_ratio: Foreground ratio [0, 1].
            k: Number of characters (optional).

        """
        self._fg_ratios.append(fg_ratio)
        if k is not None:
            self._k_counts.append(k)
        self._sample_count += 1

    def reset(self) -> None:
        """Reset all recorded statistics."""
        self._fg_ratios = []
        self._k_counts = []
        self._sample_count = 0

    def get_k0_ratio(self) -> float:
        """Get ratio of negative examples (k=0).

        Returns:
            Ratio of samples with k=0.

        """
        if not self._k_counts:
            return 0.0
        return sum(1 for k in self._k_counts if k == 0) / len(self._k_counts)

    def get_fg_ratio_stats(self) -> dict[str, float]:
        """Get foreground ratio statistics.

        Returns:
            Dictionary with mean, std, min, max of fg ratios.

        """
        if not self._fg_ratios:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        fg_tensor = torch.tensor(self._fg_ratios)
        return {
            "mean": float(fg_tensor.mean().item()),
            "std": float(fg_tensor.std().item()),
            "min": float(fg_tensor.min().item()),
            "max": float(fg_tensor.max().item()),
        }

    def get_k_distribution(self) -> dict[int, float]:
        """Get character count distribution.

        Returns:
            Dictionary mapping k to proportion.

        """
        if not self._k_counts:
            return {}

        counts: dict[int, int] = {}
        for k in self._k_counts:
            counts[k] = counts.get(k, 0) + 1

        total = len(self._k_counts)
        return {k: v / total for k, v in sorted(counts.items())}

    def save_stats(self, epoch: int | None = None) -> Path:
        """Save current statistics to JSON file.

        Args:
            epoch: Optional epoch number for filename.

        Returns:
            Path to saved stats file.

        """
        suffix = f"_epoch{epoch}" if epoch is not None else ""
        path = self.log_dir / f"distribution_stats{suffix}.json"

        stats = {
            "sample_count": self._sample_count,
            "fg_ratio_stats": self.get_fg_ratio_stats(),
            "k_distribution": self.get_k_distribution(),
            "k0_ratio": self.get_k0_ratio(),
        }

        with path.open("w") as f:
            json.dump(stats, f, indent=2)

        logger.info("Saved distribution stats to %s", path)
        return path

    def save_histogram(self, epoch: int | None = None) -> Path | None:
        """Save foreground ratio histogram.

        Args:
            epoch: Optional epoch number for filename.

        Returns:
            Path to saved histogram file, or None if no data.

        """
        if not self._fg_ratios:
            return None

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for histogram generation")
            return None

        suffix = f"_epoch{epoch}" if epoch is not None else ""
        path = self.log_dir / f"fg_ratio_histogram{suffix}.png"

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self._fg_ratios, bins=self.bins, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Foreground Ratio")
        ax.set_ylabel("Count")
        ax.set_title(f"Foreground Ratio Distribution (n={len(self._fg_ratios)})")

        # Add statistics annotation
        stats = self.get_fg_ratio_stats()
        stats_text = f"Mean: {stats['mean']:.3f}\nStd: {stats['std']:.3f}"
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved histogram to %s", path)
        return path


class SynthesisStatsTracker:
    """Track statistics during synthesis for debugging and monitoring.

    Provides detailed tracking of synthesis process including timing,
    success rates, and distribution matching.
    """

    def __init__(self) -> None:
        """Initialize stats tracker."""
        self._synthesis_count = 0
        self._placement_failures = 0
        self._blending_methods: dict[str, int] = {}
        self._k_requested: list[int] = []
        self._k_placed: list[int] = []

    def record_synthesis(
        self,
        k_requested: int,
        k_placed: int,
        blending_method: str | None = None,
    ) -> None:
        """Record a synthesis operation.

        Args:
            k_requested: Number of characters requested.
            k_placed: Number actually placed.
            blending_method: Name of blending method used.

        """
        self._synthesis_count += 1
        self._k_requested.append(k_requested)
        self._k_placed.append(k_placed)

        if k_placed < k_requested:
            self._placement_failures += k_requested - k_placed

        if blending_method:
            self._blending_methods[blending_method] = (
                self._blending_methods.get(blending_method, 0) + 1
            )

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary of summary statistics.

        """
        if not self._k_requested:
            return {"synthesis_count": 0}

        return {
            "synthesis_count": self._synthesis_count,
            "placement_failure_rate": self._placement_failures / max(1, sum(self._k_requested)),
            "blending_distribution": self._blending_methods.copy(),
            "avg_k_requested": sum(self._k_requested) / len(self._k_requested),
            "avg_k_placed": sum(self._k_placed) / len(self._k_placed),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self._synthesis_count = 0
        self._placement_failures = 0
        self._blending_methods = {}
        self._k_requested = []
        self._k_placed = []
