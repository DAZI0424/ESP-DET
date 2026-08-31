"""Build a deterministic, class-balanced ESP-DL calibration image set."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_dataset(dataset_yaml: Path, split: str) -> tuple[Path, Path, dict[int, str]]:
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", dataset_yaml.parent))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    image_dir = root / config[split]
    parts = list(image_dir.parts)
    try:
        images_index = parts.index("images")
    except ValueError as error:
        raise ValueError(f"Split path must contain an 'images' directory: {image_dir}") from error
    parts[images_index] = "labels"
    label_dir = Path(*parts)
    names_raw = config["names"]
    if isinstance(names_raw, list):
        names = dict(enumerate(names_raw))
    else:
        names = {int(key): value for key, value in names_raw.items()}
    return image_dir, label_dir, names


def parse_labels(path: Path) -> list[tuple[int, float]]:
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(float(fields[0]))
        area = float(fields[3]) * float(fields[4])
        labels.append((class_id, area))
    return labels


def spread_sample(items: list[tuple[float, Path]], count: int, excluded: set[Path]) -> list[Path]:
    candidates = [item for item in sorted(items, key=lambda item: (item[0], item[1].name)) if item[1] not in excluded]
    if len(candidates) < count:
        raise ValueError(f"Need {count} unique candidates but only {len(candidates)} remain")
    if count == 1:
        return [candidates[len(candidates) // 2][1]]
    positions = [round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)]
    return [candidates[position][1] for position in positions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--background", type=int, default=32)
    args = parser.parse_args()

    image_dir, label_dir, names = load_dataset(args.dataset.resolve(), args.split)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    by_class: dict[int, list[tuple[float, Path]]] = defaultdict(list)
    backgrounds: list[tuple[float, Path]] = []
    all_images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in all_images:
        labels = parse_labels(label_dir / f"{image_path.stem}.txt")
        if not labels:
            backgrounds.append((0.0, image_path))
            continue
        areas_by_class: dict[int, list[float]] = defaultdict(list)
        for class_id, area in labels:
            areas_by_class[class_id].append(area)
        for class_id, areas in areas_by_class.items():
            by_class[class_id].append((max(areas), image_path))

    selected: list[tuple[str, int | None, Path]] = []
    used: set[Path] = set()
    for class_id in sorted(names):
        paths = spread_sample(by_class[class_id], args.per_class, used)
        for path in paths:
            used.add(path)
            selected.append((f"class_{class_id}", class_id, path))
    for path in spread_sample(backgrounds, args.background, used):
        used.add(path)
        selected.append(("background", None, path))

    manifest_items = []
    for index, (group, class_id, source) in enumerate(selected):
        destination_name = f"{index:03d}_{group}_{source.name}"
        destination = args.output / destination_name
        shutil.copy2(source, destination)
        manifest_items.append(
            {
                "file": destination_name,
                "group": group,
                "class_id": class_id,
                "source": str(source),
            }
        )

    manifest = {
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "selection": "class-balanced; per-class candidates spread by normalized box area",
        "per_class": args.per_class,
        "background": args.background,
        "total": len(manifest_items),
        "classes": names,
        "group_counts": dict(Counter(item["group"] for item in manifest_items)),
        "images": manifest_items,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in ("total", "group_counts")}, indent=2))


if __name__ == "__main__":
    main()
