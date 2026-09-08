"""Measure M013 group scale only; exports no image/animation frame data."""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
source = next((ROOT/'left/left').glob('M013*.mp4'))
cap = cv2.VideoCapture(str(source))
assert cap.get(cv2.CAP_PROP_FPS) == 60
keys = []
for frame in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ok, bgr = cap.read()
    assert ok
    category = 0 if frame < 20 else 1 if frame < 116 else 2 if frame < 200 else 3 if frame < 285 else 4
    ys, xs = np.where(bgr[:,445:].max(axis=2) > 80)
    if len(xs):
        full_height = 184 if category in (2,4) else 185
        scale = 0.4 * (xs.max()-xs.min()) / 178 + 0.6 * (ys.max()-ys.min()) / full_height
        scale = 1 if scale >= 0.985 else scale
        scale = min(256, max(1, round(scale * 256)))
    else:
        scale = 0
    keys.append((category, scale))
cap.release()
text = '// M013: 307 samples at 60 fps. Category index + LVGL scale; no image pixels.\n#pragma once\n'
text += '#include <stdint.h>\ntypedef struct { uint16_t scale; uint8_t category; } ui_category_key_t;\n'
text += 'static const ui_category_key_t category_keys[307] = {\n'
text += '\n'.join('    {'+str(scale)+','+str(category)+'}, // '+str(frame) for frame,(category,scale) in enumerate(keys))
text += '\n};\n'
(ROOT/'main/ui/ui_category_timing.h').write_text(text)
print('Generated M013 scale timeline: 307 samples; 5117 ms; final state held.')
