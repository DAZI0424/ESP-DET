"""Compare installed MiSans weights against the MP4's first-frame text masks."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
fonts = Path.home() / 'AppData/Local/Microsoft/Windows/Fonts'
cap = cv2.VideoCapture(str(root / '886ee293382508b063daf651dbeac3d8.mp4'))
ok, bgr = cap.read()
cap.release()
assert ok
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
samples = {
    'total_title': ('总计', (218, 0, 288, 36), range(22, 33)),
    'weight_title': ('重量', (218, 52, 288, 92), range(22, 33)),
    'calorie_title': ('热量', (461, 52, 536, 92), range(22, 33)),
    'total_value': ('0', (374, 0, 413, 37), range(22, 34)),
    'grams_unit': ('g', (375, 52, 415, 99), range(22, 34)),
    'kcal_unit': ('kcal', (576, 52, 647, 97), range(22, 34)),
    'calorie_total': ('0', (615, 0, 648, 37), range(22, 34)),
    'calorie_value': ('0', (581, 109, 648, 199), range(76, 97)),
    'weight_value': ('0', (349, 109, 419, 199), range(76, 97)),
}
result = {}
for name, (text, (x1, y1, x2, y2), sizes) in samples.items():
    target = rgb[y1:y2, x1:x2].max(axis=2).astype(np.float32) / 255
    candidates = []
    for path in fonts.glob('MiSans-*.ttf'):
        for size in sizes:
            font = ImageFont.truetype(str(path), size)
            box = font.getbbox(text)
            mask = Image.new('L', (box[2]-box[0]+4, box[3]-box[1]+4))
            ImageDraw.Draw(mask).text((2-box[0], 2-box[1]), text, font=font, fill=255)
            sample = np.array(mask).astype(np.float32) / 255
            if sample.shape[0] > target.shape[0] or sample.shape[1] > target.shape[1]:
                continue
            score = cv2.matchTemplate(target, sample, cv2.TM_CCOEFF_NORMED)
            _, peak, _, pos = cv2.minMaxLoc(score)
            candidates.append({'font': path.name, 'size': size, 'correlation': round(peak, 5),
                               'ink_x': x1+pos[0]+2, 'ink_y': y1+pos[1]+2,
                               'ink_size': [box[2]-box[0], box[3]-box[1]]})
    result[name] = sorted(candidates, key=lambda item: item['correlation'], reverse=True)[:6]
(root / 'tmp/ui-font-comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
for name, candidates in result.items():
    print(name, candidates[:3])
