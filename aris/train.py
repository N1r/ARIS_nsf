"""Training entry point (Lightning CLI).

    aris-train fit -c configs/aris_16k.yaml --data.manifest_path data/f024/manifest.csv

Outputs go to ``<trainer.default_root_dir>/<name>/version_N/`` (config, metrics, checkpoints,
validation audio). Resume with ``--ckpt_path .../last.ckpt``.
"""

import torch
from lightning.pytorch.cli import LightningCLI

from aris.data import ManifestDataModule
from aris.model import ARIS


def main():
    torch.set_float32_matmul_precision("high")
    LightningCLI(ARIS, ManifestDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
