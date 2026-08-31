"""Reproducible ESPDet-Pico training entry point for Welland V02."""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import Train  # noqa: E402


MODEL = ROOT / "cfg" / "models" / "espdet_pico.yaml"
DATASET = ROOT / "datasets" / "Welland-V02-espdet-pico" / "clean-20260810.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="welland_v02_espdet_pico_224")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = Path(args.resume).expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

    settings = dict(
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        device=0,
        workers=args.workers,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=20,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        rect=False,
        cache=False,
        amp=True,
        seed=42,
        deterministic=True,
        project=str(ROOT / "runs" / "espdet"),
        name=args.name,
        exist_ok=False,
        plots=True,
        save=True,
        val=True,
        verbose=True,
    )
    if resume_checkpoint:
        settings["resume"] = True

    print(f"MODEL_CONFIG={MODEL}", flush=True)
    print(f"DATASET_CONFIG={DATASET}", flush=True)
    results = Train(
        pretrained_path=str(resume_checkpoint) if resume_checkpoint else None,
        dataset=str(DATASET),
        imgsz=224,
        **settings,
    )
    print(f"TRAIN_SAVE_DIR={results.save_dir}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
