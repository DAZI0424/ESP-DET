"""Create non-destructive, capture-grouped train/validation lists for Welland v04."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageFile


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def sequence_key(stem: str):
    compact = re.search(r"(\d{8})[_-]?(\d{6})", stem)
    if compact:
        hhmmss = compact.group(2)
        seconds = int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:])
        family = stem[: compact.start()].lower() + compact.group(1)
        return family, seconds, 300
    separated = re.search(r"(\d{8})[_-](\d{2})[_-](\d{2})[_-](\d{2})", stem)
    if separated:
        seconds = int(separated.group(2)) * 3600 + int(separated.group(3)) * 60 + int(separated.group(4))
        family = stem[: separated.start()].lower() + separated.group(1)
        return family, seconds, 300
    numeric = re.fullmatch(r"(?P<prefix>img_)?(?P<number>\d+)", stem, re.IGNORECASE)
    if numeric:
        return (numeric.group("prefix") or "numeric").lower(), int(numeric.group("number")), 20
    return None


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return value


def label_counts(image: Path) -> Counter:
    parts = list(image.parts)
    parts[parts.index("images")] = "labels"
    label = Path(*parts).with_suffix(".txt")
    counts = Counter()
    if label.exists():
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                counts[int(fields[0])] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/welland-v04-espdet-pico/welland-v04-local.yaml"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(config["path"]).resolve()
    images = sorted(
        {
            path.resolve()
            for split in ("train", "val")
            for path in (root / "images" / split).iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        }
    )
    union = UnionFind(len(images))

    sequences = defaultdict(list)
    for index, image in enumerate(images):
        key = sequence_key(image.stem)
        if key:
            family, order, maximum_gap = key
            sequences[(family, maximum_gap)].append((order, index))
    for (_, maximum_gap), members in sequences.items():
        members.sort()
        for (previous, left), (current, right) in zip(members, members[1:]):
            if current - previous <= maximum_gap:
                union.union(left, right)

    exact, visual = {}, {}
    exact_duplicate_links = 0
    visual_duplicate_links = 0
    for index, image in enumerate(images):
        digest = hashlib.sha256(image.read_bytes()).digest()
        if digest in exact:
            union.union(index, exact[digest])
            exact_duplicate_links += 1
        else:
            exact[digest] = index
        dhash = difference_hash(image)
        if dhash in visual:
            union.union(index, visual[dhash])
            visual_duplicate_links += 1
        else:
            visual[dhash] = index

    grouped = defaultdict(list)
    for index, image in enumerate(images):
        grouped[union.find(index)].append(image)
    groups = list(grouped.values())
    group_counts = [sum((label_counts(image) for image in group), Counter()) for group in groups]
    total_counts = sum(group_counts, Counter())
    target_counts = {class_id: count * args.val_fraction for class_id, count in total_counts.items()}
    target_images = len(images) * args.val_fraction

    rng = random.Random(args.seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    order.sort(key=lambda index: len(groups[index]), reverse=True)
    val_groups, val_counts, val_images = set(), Counter(), 0

    def cost(counts: Counter, image_count: int) -> float:
        class_cost = sum(
            abs(counts[class_id] - target) / max(target, 1)
            for class_id, target in target_counts.items()
        )
        return class_cost + abs(image_count - target_images) / max(target_images, 1)

    for index in order:
        candidate_counts = val_counts + group_counts[index]
        candidate_images = val_images + len(groups[index])
        if cost(candidate_counts, candidate_images) < cost(val_counts, val_images):
            val_groups.add(index)
            val_counts = candidate_counts
            val_images = candidate_images

    train = sorted(image for index, group in enumerate(groups) if index not in val_groups for image in group)
    val = sorted(image for index, group in enumerate(groups) if index in val_groups for image in group)
    assert train and val and set(train).isdisjoint(val)

    train_list = root / "train_grouped.txt"
    val_list = root / "val_grouped.txt"
    train_list.write_text("\n".join(map(str, train)) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(map(str, val)) + "\n", encoding="utf-8")
    output_yaml = root / "welland-v04-grouped.yaml"
    output_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": train_list.name,
                "val": val_list.name,
                "names": config["names"],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = {
        "images": len(images),
        "groups": len(groups),
        "train_images": len(train),
        "val_images": len(val),
        "exact_duplicate_links": exact_duplicate_links,
        "visual_duplicate_links": visual_duplicate_links,
        "total_instances": dict(sorted(total_counts.items())),
        "val_instances": dict(sorted(val_counts.items())),
        "yaml": str(output_yaml),
    }
    (root / "grouped_split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
