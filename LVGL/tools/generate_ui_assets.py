"""Generate MiSans subsets and ONLY the permitted eyes/apple picture placeholders.

No screen captures, video frames, text bitmaps or animation packets are linked
into the product UI. Text is generated from font glyphs by LVGL at runtime.
Requires numpy, opencv-python, fonttools and lv_font_conv 1.5.3.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import cv2
from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--font-dir', type=Path, default=Path.home() / 'AppData/Local/Microsoft/Windows/Fonts')
args = parser.parse_args()
out = ROOT / 'main/ui'
(out / 'fonts/source').mkdir(parents=True, exist_ok=True)
(out / 'assets').mkdir(exist_ok=True)
converter = ROOT / 'tmp/ui-node-tools/node_modules/lv_font_conv/lv_font_conv.js'
roles = [
    ('demibold_27', 'Demibold', 27, '总计热量蛋白质脂肪碳水化合物钠'),
    ('medium_27', 'Medium', 27, '重量'),
    ('normal_28', 'Normal', 28, '0123456789.- '),
    ('normal_27', 'Normal', 27, 'mgkcal '),
    ('light_85', 'Light', 85, '0123456789.- '),
    ('light_84', 'Light', 84, '0123456789.- '),
    ('demibold_28', 'Demibold', 28, '苹果'),
    ('demibold_36', 'Demibold', 36, '苹果'),
]
metadata = {'reference': 'MP4 first frame; MiSans family confirmed by user; weights fitted to reference', 'fonts': [], 'images': []}
for name, weight, size, symbols in roles:
    original = args.font_dir / f'MiSans-{weight}.ttf'
    font = TTFont(original)
    original_weight = font['OS/2'].usWeightClass
    options = subset.Options()
    options.name_IDs = ['*']
    sub = subset.Subsetter(options=options)
    sub.populate(text=symbols)
    sub.subset(font)
    source = out / 'fonts/source' / f'{name}.ttf'
    font.save(source)
    target = out / 'fonts' / f'ui_font_{name}.c'
    subprocess.run(['node', str(converter), '--font', str(source), '--size', str(size),
                    '--bpp', '4', '--format', 'lvgl', '--symbols', symbols,
                    '--no-compress', '--lv-font-name', f'ui_font_{name}',
                    '-o', str(target)], check=True)
    # Avoid leaking developer-machine paths in generated source comments.
    text = target.read_text(encoding='utf-8').replace(str(ROOT), '<LVGL>')
    target.write_text(text, encoding='utf-8')
    metadata['fonts'].append({'name': name, 'family': 'MiSans', 'weight': weight,
                              'source_weight_class': original_weight, 'size_px': size,
                              'symbols': symbols, 'source_sha256': hashlib.sha256(original.read_bytes()).hexdigest()})

video = ROOT / '886ee293382508b063daf651dbeac3d8.mp4'
capture = cv2.VideoCapture(str(video))
for name, frame, box in [('eyes', 0, (0, 38, 136, 154)), ('apple', 60, (20, 19, 116, 113))]:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    assert ok
    x1, y1, x2, y2 = box
    pixels = image[y1:y2, x1:x2].copy()
    cv2.imwrite(str(out / 'assets' / f'{name}.png'), pixels)
    data = pixels.tobytes()
    lines = [', '.join(f'0x{b:02x}' for b in data[i:i+24]) for i in range(0, len(data), 24)]
    source = '// Picture placeholder cropped from the supplied reference; no UI text.\n#include "lvgl.h"\n'
    source += f'static const uint8_t {name}_pixels[] = {{\n' + ',\n'.join(lines) + '\n};\n'
    source += f'const lv_image_dsc_t ui_asset_{name} = {{\n'
    source += f'    .header = {{.magic = LV_IMAGE_HEADER_MAGIC, .cf = LV_COLOR_FORMAT_RGB888, .w = {x2-x1}, .h = {y2-y1}, .stride = {(x2-x1)*3}}},\n'
    source += f'    .data_size = sizeof({name}_pixels), .data = {name}_pixels,\n}};\n'
    (out / 'assets' / f'ui_asset_{name}.c').write_text(source, encoding='utf-8')
    metadata['images'].append({'name': name, 'source_frame': frame, 'crop_xyxy': box, 'bytes': len(data)})
capture.release()
(out / 'assets/PROVENANCE.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(f'Generated {len(roles)} MiSans font roles and 2 picture placeholders; no full-frame image asset')
