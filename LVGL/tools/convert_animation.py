"""Convert the supplied video to lossless BGR888 temporal patches for LVGL 9.

Host dependencies: numpy, opencv-python. No resizing, frame dropping or palette.
Each frame: uint32 patch count; patches: uint32 pixel offset/count, BGR bytes.
Frame zero is a complete keyframe. Physical pixels are clockwise-rotated.
"""
from pathlib import Path
import hashlib
import json
import struct
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '886ee293382508b063daf651dbeac3d8.mp4'
OUT = ROOT / 'main' / 'assets'


def convert():
    OUT.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(SOURCE))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open {SOURCE}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if abs(fps - 30) > 0.001:
        raise ValueError('This asset pipeline expects the supplied 30 fps video')
    payload = bytearray()
    offsets, hashes, timestamps, decoded = [], [], [], []
    previous = None
    while True:
        ok, source = capture.read()
        if not ok:
            break
        if source.shape != (200, 648, 3):
            raise ValueError(f'Expected 648x200 video, got {source.shape}')
        timestamps.append(capture.get(cv2.CAP_PROP_POS_MSEC))
        current = np.rot90(source, -1).copy().reshape(-1, 3)
        decoded.append(current.reshape(648, 200, 3))
        hashes.append(hashlib.sha256(current.tobytes()).hexdigest())
        changed = np.arange(len(current)) if previous is None else np.flatnonzero(np.any(current != previous, axis=1))
        # Merge small gaps to reduce packet overhead without changing any pixels.
        starts = changed[np.r_[True, np.diff(changed) > 8]] if len(changed) else []
        ends = changed[np.r_[np.diff(changed) > 8, True]] + 1 if len(changed) else []
        offsets.append(len(payload))
        payload.extend(struct.pack('<I', len(starts)))
        for start, end in zip(starts, ends):
            payload.extend(struct.pack('<II', int(start), int(end - start)))
            payload.extend(current[start:end].tobytes())
        previous = current
    capture.release()
    if len(offsets) != 110:
        raise ValueError(f'Expected 110 decoded frames, got {len(offsets)}')
    if not np.allclose(timestamps, np.arange(len(offsets)) * 1000 / fps, atol=0.2):
        raise ValueError('Video is not constant frame rate; preserve PTS before conversion')
    offsets.append(len(payload))
    # Extra packet: last -> first, so looping need not copy the entire startup
    # keyframe from flash into PSRAM. Preserve all 24 bits, not just RGB666.
    first = decoded[0].reshape(-1, 3)
    last = decoded[-1].reshape(-1, 3)
    changed = np.flatnonzero(np.any(first != last, axis=1))
    starts = changed[np.r_[True, np.diff(changed) > 8]] if len(changed) else []
    ends = changed[np.r_[np.diff(changed) > 8, True]] + 1 if len(changed) else []
    payload.extend(struct.pack('<I', len(starts)))
    for start, end in zip(starts, ends):
        payload.extend(struct.pack('<II', int(start), int(end - start)))
        payload.extend(first[start:end].tobytes())
    offsets.append(len(payload))
    (OUT / 'animation.bin').write_bytes(payload)
    rectangles, rectangle_offsets = [], []
    # Frame 0's dirty regions compare the last frame with the first. Startup
    # separately refreshes the full screen before entering the loop.
    for frame, current in enumerate(decoded):
        rectangle_offsets.append(len(rectangles))
        previous_frame = decoded[frame - 1]
        regions = []
        for y in range(0, 648, 8):
            changed = np.any(((current[y:y+8] ^ previous_frame[y:y+8]) & 252) != 0, axis=2)
            xs = np.flatnonzero(np.any(changed, axis=0))
            if len(xs):
                starts = xs[np.r_[True, np.diff(xs) > 8]]
                ends = xs[np.r_[np.diff(xs) > 8, True]]
                regions.extend((int(left), y, int(right), min(y + 7, 647))
                               for left, right in zip(starts, ends))
        # Rendering many tiny LVGL regions costs more than a few extra unchanged
        # pixels. Greedily merge nearby rectangles; all added pixels are copied
        # from the exact current frame, never approximated or interpolated.
        regions = np.array(regions, dtype=np.int32).reshape(-1, 4)
        while len(regions) > 1:
            area = (regions[:, 2] - regions[:, 0] + 1) * (regions[:, 3] - regions[:, 1] + 1)
            left = np.minimum(regions[:, None, 0], regions[None, :, 0])
            top = np.minimum(regions[:, None, 1], regions[None, :, 1])
            right = np.maximum(regions[:, None, 2], regions[None, :, 2])
            bottom = np.maximum(regions[:, None, 3], regions[None, :, 3])
            extra = (right - left + 1) * (bottom - top + 1) - area[:, None] - area[None, :]
            extra[np.tril_indices(len(regions))] = 999999
            i, j = np.unravel_index(extra.argmin(), extra.shape)
            if extra[i, j] > 1000:
                break
            regions[i] = [left[i, j], top[i, j], right[i, j], bottom[i, j]]
            regions = np.delete(regions, j, axis=0)
        rectangles.extend(tuple(map(int, rect)) for rect in regions)
    rectangle_offsets.append(len(rectangles))
    (OUT / 'animation_index.h').write_text(
        '// Generated by tools/convert_animation.py; do not edit.\n#pragma once\n#include <stdint.h>\n'
        '#define ANIMATION_WIDTH 200U\n#define ANIMATION_HEIGHT 648U\n'
        f'#define ANIMATION_FRAME_COUNT {len(hashes)}U\n#define ANIMATION_FPS 30U\n'
        f'#define ANIMATION_LOOP_PACKET {len(hashes)}U\n'
        'static const uint32_t animation_offsets[] = {\n    ' +
        ', '.join(str(n) + 'U' for n in offsets) + '\n};\n'
        'typedef struct { uint16_t x1, y1, x2, y2; } animation_rect_t;\n'
        'static const uint32_t animation_rect_offsets[] = {\n    ' +
        ', '.join(str(n) + 'U' for n in rectangle_offsets) + '\n};\n'
        'static const animation_rect_t animation_rects[] = {\n' +
        ''.join('    {' + ', '.join(map(str, rect)) + '},\n' for rect in rectangles) +
        '};\n', encoding='utf-8')
    (OUT / 'animation.json').write_text(json.dumps({
        'source': SOURCE.name, 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'decoder': 'OpenCV ' + cv2.__version__, 'source_size': [648, 200],
        'panel_size': [200, 648], 'rotation': '90 degrees clockwise; no scaling',
        'fps': fps, 'frames': len(hashes), 'duration_ms': len(hashes) * 1000 / fps,
        'format': 'BGR888 temporal patches, little endian uint32, lossless after MP4 decode',
        'asset_bytes': len(payload), 'asset_sha256': hashlib.sha256(payload).hexdigest(),
        'frame_sha256': hashes,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Converted {len(hashes)} frames, {len(payload):,} bytes, {len(hashes)/fps:.6f} seconds')


if __name__ == '__main__':
    convert()
