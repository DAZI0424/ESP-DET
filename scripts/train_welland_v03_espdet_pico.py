"""Train ESPDet-Pico on the Welland v03 dataset using the previous recipe."""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import Train  # noqa: E402


if __name__ == "__main__":
    multiprocessing.freeze_support()
    Train(
        dataset=str(ROOT / "datasets" / "welland-v03-espdet-pico" / "data.yaml"),
        epochs=500,
        patience=75,
        imgsz=224,
        device=0,
        batch=-1,
        workers=8,
        cache=False,
        amp=True,
        cos_lr=True,
        project=str(ROOT / "runs" / "detect"),
        name="welland_v03_espdet_pico_500e_gpu",
        exist_ok=False,
        save=True,
        save_period=25,
        plots=True,
        verbose=True,
        seed=0,
    )
