from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from .data_module import AnimeSegDataModule
from .lit_module import AnimeSegmentation


def main() -> None:
    LightningCLI(
        AnimeSegmentation,
        AnimeSegDataModule,
        seed_everything_default=7,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
