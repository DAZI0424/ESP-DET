"""Reproducible ESPDet-Pico training entry point for Welland-V0.1."""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import Train  # noqa: E402


DATASET = ROOT / "datasets" / "welland-v01-espdet-pico" / "welland-v01.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    # Keep the Windows process count and RAM usage bounded. Ultralytics may
    # create additional validation workers on top of this value.
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="welland_v01_espdet_pico_224")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = Path(args.resume).expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

    common = dict(
        epochs=1 if args.probe else args.epochs,
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
        exist_ok=args.probe,
        plots=not args.probe,
        save=not args.probe,
        val=not args.probe,
        verbose=True,
    )
    if args.probe:
        common["fraction"] = 0.1
    if resume_checkpoint:
        # Ultralytics expects the checkpoint to be loaded into YOLO first and
        # resume=True afterwards. Passing a path while starting from the model
        # YAML is discarded by Model.train() in Ultralytics 8.4.x.
        common["resume"] = True

    results = Train(
        pretrained_path=str(resume_checkpoint) if resume_checkpoint else None,
        dataset=str(DATASET),
        imgsz=224,
        **common,
    )
    print(f"TRAIN_SAVE_DIR={results.save_dir}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
