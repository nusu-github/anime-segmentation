"""Test split management for evaluation protocols.

Provides TestSplit and TestSplitManager for managing separate evaluation
datasets with different characteristics (standard, stress, negative).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class TestSplit:
    """A named test split with associated metadata.

    Attributes:
        name: Unique identifier for the split.
        description: Human-readable description of the split.
        image_paths: List of image file paths.
        mask_paths: List of corresponding mask file paths.
        metadata: Additional metadata (e.g., split type, criteria).

    """

    name: str
    description: str
    image_paths: list[str]
    mask_paths: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.image_paths)


class TestSplitManager:
    """Manager for test split creation and loading.

    Supports creating, saving, and loading test splits with different
    characteristics for comprehensive evaluation.

    Split types:
        - standard: Normal test images with typical characteristics
        - stress: Challenging cases (complex backgrounds, occlusion, etc.)
        - negative: Images without foreground objects (for FP rate testing)
    """

    def __init__(self, splits_dir: str | Path) -> None:
        """Initialize test split manager.

        Args:
            splits_dir: Directory for storing split definition files.

        """
        self.splits_dir = Path(splits_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def _split_path(self, name: str) -> Path:
        """Get path for split definition file."""
        return self.splits_dir / f"{name}.json"

    def load_split(self, name: str) -> TestSplit:
        """Load a test split by name.

        Args:
            name: Name of the split to load.

        Returns:
            Loaded TestSplit instance.

        Raises:
            FileNotFoundError: If split definition file does not exist.

        """
        path = self._split_path(name)
        if not path.exists():
            msg = f"Split definition not found: {path}"
            raise FileNotFoundError(msg)

        with path.open("r") as f:
            data = json.load(f)

        return TestSplit(
            name=data["name"],
            description=data.get("description", ""),
            image_paths=data["image_paths"],
            mask_paths=data["mask_paths"],
            metadata=data.get("metadata", {}),
        )

    def save_split(self, split: TestSplit) -> Path:
        """Save a test split definition.

        Args:
            split: TestSplit instance to save.

        Returns:
            Path to the saved split file.

        """
        path = self._split_path(split.name)

        data = {
            "name": split.name,
            "description": split.description,
            "image_paths": split.image_paths,
            "mask_paths": split.mask_paths,
            "metadata": split.metadata,
        }

        with path.open("w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved split '%s' with %d samples to %s", split.name, len(split), path)
        return path

    def create_split(
        self,
        name: str,
        image_paths: list[str],
        mask_paths: list[str],
        criteria: Callable[[Path], bool] | None = None,
        description: str = "",
        split_type: Literal["standard", "stress", "negative"] = "standard",
    ) -> TestSplit:
        """Create a new test split.

        Args:
            name: Unique name for the split.
            image_paths: List of image file paths.
            mask_paths: List of corresponding mask file paths.
            criteria: Optional filter function applied to image paths.
            description: Human-readable description.
            split_type: Type of split (standard, stress, negative).

        Returns:
            Created TestSplit instance (also saved to disk).

        """
        # Apply criteria filter if provided
        if criteria is not None:
            filtered_images = []
            filtered_masks = []
            for img, mask in zip(image_paths, mask_paths, strict=True):
                if criteria(Path(img)):
                    filtered_images.append(img)
                    filtered_masks.append(mask)
            image_paths = filtered_images
            mask_paths = filtered_masks

        split = TestSplit(
            name=name,
            description=description,
            image_paths=image_paths,
            mask_paths=mask_paths,
            metadata={"split_type": split_type},
        )

        self.save_split(split)
        return split

    def list_splits(self) -> list[str]:
        """List all available split names.

        Returns:
            List of split names.

        """
        return [p.stem for p in self.splits_dir.glob("*.json")]

    @staticmethod
    def infer_split_type(
        image_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Literal["standard", "stress", "negative"]:
        """Infer split type from image path or metadata.

        Args:
            image_path: Path to the image file.
            metadata: Optional metadata dict with hints.

        Returns:
            Inferred split type.

        """
        if metadata is not None:
            if metadata.get("is_negative", False):
                return "negative"
            if metadata.get("is_stress", False):
                return "stress"

        # Heuristic: check path components
        path_str = str(image_path).lower()
        if "negative" in path_str or "bg_only" in path_str:
            return "negative"
        if "stress" in path_str or "hard" in path_str or "complex" in path_str:
            return "stress"

        return "standard"
