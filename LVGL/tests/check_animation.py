"""Independent packet decode and panel-transfer verification against source MP4."""
from pathlib import Path
import hashlib
import json
import re
import struct
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
assets = root / 'main/assets'
meta = json.loads((assets / 'animation.json').read_text(encoding='utf-8'))
blob = (assets / 'animation.bin').read_bytes()
index = (assets / 'animation_index.h').read_text()
offsets = [int(x) for x in re.findall(r'(\d+)U', index.split('animation_offsets[]')[1].split('};')[0])]
rect_offsets = [int(x) for x in re.findall(r'(\d+)U', index.split('animation_rect_offsets[]')[1].split('};')[0])]
rects = [tuple(map(int, match)) for match in re.findall(r'\{(\d+), (\d+), (\d+), (\d+)\}', index)]
assert hashlib.sha256(blob).hexdigest() == meta['asset_sha256']
assert hashlib.sha256((root / meta['source']).read_bytes()).hexdigest() == meta['source_sha256']
assert len(offsets) == meta['frames'] + 2 and offsets[0] == 0 and offsets[-1] == len(blob)

canvas = bytearray(200 * 648 * 3)
panel = np.zeros((648, 200, 3), dtype=np.uint8)
previous = panel.copy()
capture = cv2.VideoCapture(str(root / meta['source']))
frames, band_pixels = [], []

def decode(frame):
    pos, limit = offsets[frame], offsets[frame + 1]
    count, = struct.unpack_from('<I', blob, pos)
    pos += 4
    for _ in range(count):
        start, size = struct.unpack_from('<II', blob, pos)
        pos += 8
        assert start + size <= 200 * 648 and pos + size * 3 <= limit
        canvas[start * 3:(start + size) * 3] = blob[pos:pos + size * 3]
        pos += size * 3
    assert pos == limit
    return np.frombuffer(canvas, np.uint8).reshape(648, 200, 3)

for frame in range(meta['frames'] + 1):
    current = decode(frame)
    if frame < meta['frames']:
        ok, source = capture.read()
        assert ok
        expected = np.rot90(source, -1).copy()
        assert np.array_equal(current, expected), f'Pixel mismatch in frame {frame}'
        assert hashlib.sha256(canvas).hexdigest() == meta['frame_sha256'][frame]
        if frame in (0, 30, 40, 60, 109):
            frames.append(source)
    else:
        # Last -> first loop must restore a keyframe, without old apple pixels.
        assert hashlib.sha256(canvas).hexdigest() == meta['frame_sha256'][0]
    transferred = 0
    f = frame % meta['frames']
    regions = [(0, 0, 199, 647)] if frame == 0 else rects[rect_offsets[f]:rect_offsets[f + 1]]
    for left, top, right, bottom in regions:
        assert 0 <= left <= right < 200 and 0 <= top <= bottom < 648
        panel[top:bottom+1, left:right+1] = current[top:bottom+1, left:right+1, ::-1] & 252
        transferred += (right - left + 1) * (bottom - top + 1)
    assert np.array_equal(panel, current[:, :, ::-1] & 252), f'Panel diff mismatch {frame}'
    previous[:] = current
    if frame:
        band_pixels.append(transferred)
assert not capture.read()[0], 'Unexpected extra source frames'
capture.release()
preview = root / 'tmp/animation-verification.png'
preview.parent.mkdir(exist_ok=True)
cv2.imwrite(str(preview), np.concatenate(frames, axis=0))
print(f'PASS: all {meta["frames"]} frames match decoded MP4 pixel-for-pixel; loop keyframe restores exactly')
print('PASS: differential bands reproduce every RGB666 panel pixel including loop transition')
print(f'Asset {len(blob):,} bytes; canvas {len(canvas):,} bytes; max transfer {max(band_pixels):,} pixels/frame')
print(f'40 MHz worst pixel-only wire time {max(band_pixels)*24/40000:.3f} ms (not hardware FPS)')
