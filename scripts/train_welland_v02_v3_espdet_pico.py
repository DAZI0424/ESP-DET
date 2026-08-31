"""Reproducible training entry point for the Welland v02-v3 dataset."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import Train


if __name__ == "__main__":
    Train(
        dataset=str(ROOT / "datasets" / "welland-v02-v3-yolo" / "data.yaml"),
        epochs=500,
        patience=75,
        imgsz=224,
        device=0,
        batch=-1,  # Ultralytics auto-batch: targets 60% of available GPU memory.
        workers=8,
        cache=False,  # The source images are larger than practical for 16 GB system RAM.
        amp=True,
        cos_lr=True,
        project=str(ROOT / "runs" / "detect"),
        name="welland_v02_v3_espdet_pico_500e_gpu",
        exist_ok=False,
        save=True,
        save_period=25,
        plots=True,
        verbose=True,
        seed=0,
    )
