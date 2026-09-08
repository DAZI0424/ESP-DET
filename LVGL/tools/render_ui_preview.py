"""Build/run the real LVGL C widgets on Windows using zig cc; no UI reimplementation."""
from pathlib import Path
import argparse
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--cc', type=Path, default=ROOT/'tmp/host-tools/ziglang/zig.exe')
args = parser.parse_args()
sources = sorted((ROOT/'managed_components/lvgl__lvgl/src').rglob('*.c'))
sources += sorted((ROOT/'main/ui').rglob('*.c')) + [ROOT/'tests/ui_preview/main.c']
flags = ['-std=c11', '-O1','-UNDEBUG', '-DLV_CONF_INCLUDE_SIMPLE', '-DLV_LVGL_H_INCLUDE_SIMPLE']
flags += ['-I'+str(ROOT/p) for p in ['tests/ui_preview', 'managed_components/lvgl__lvgl', 'main/ui']]
flags += [str(p) for p in sources] + ['-o', str(ROOT/'tmp/ui_preview.exe')]
response = ROOT/'tmp/ui_preview.rsp'
response.write_text(subprocess.list2cmdline(flags))
subprocess.run([str(args.cc), 'cc', '@'+str(response)], check=True)
out = ROOT/'docs/ui'
out.mkdir(parents=True, exist_ok=True)
subprocess.run([str(ROOT/'tmp/ui_preview.exe'), str(ROOT/'tmp/ui-first.ppm'), str(ROOT/'tmp/ui-apple.ppm')], check=True)
for name in ['first', 'apple']:
    Image.open(ROOT/f'tmp/ui-{name}.ppm').save(out/f'native-{name}.png')
first = Image.open(out/'native-first.png').convert('RGB')
apple = Image.open(out/'native-apple.png').convert('RGB')
assert first.crop((200, 0, 648, 200)).tobytes() == apple.crop((200, 0, 648, 200)).tobytes(), 'Metric panels changed across variants'
for box in [(224, 3, 280, 30), (224, 61, 280, 88), (467, 3, 523, 30), (467, 61, 523, 88)]:
    region = first.crop(box)
    ink = [region.getpixel((x, y)) for y in range(region.height) for x in range(region.width)
           if max(region.getpixel((x, y))) > 50]
    assert len(ink) > 200, 'Missing title glyphs'
    assert sum(max(rgb)-min(rgb) > 40 for rgb in ink) > 150, 'Missing title color'
print('PASS: all four colored titles present; metric panels identical across picture variants.')
print('Preview images: '+str(out))
motion = [Image.open(ROOT/f'tmp/ui-first.ppm.loading-{i:02d}.ppm').convert('RGB') for i in range(30)]
motion[0].save(out/'native-loading.gif', save_all=True, append_images=motion[1:],
               duration=50, loop=0, disposal=2)
print('Animated native LVGL preview: '+str(out/'native-loading.gif'))
