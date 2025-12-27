"""Combined dataset wrapper for real and synthetic datasets."""

from torch.utils.data import ConcatDataset, Dataset
from torchvision import tv_tensors


class CombinedDataset(ConcatDataset):
    """Combined dataset of real and synthetic samples.

    Wraps multiple datasets (typically RealImageDataset and SyntheticDataset)
    into a single dataset using ConcatDataset.

    Args:
        datasets: List of datasets to combine.
    """

    def __init__(self, datasets: list[Dataset]) -> None:
        super().__init__(datasets)

    def __getitem__(self, idx: int) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        # ConcatDataset already handles index mapping
        return super().__getitem__(idx)
