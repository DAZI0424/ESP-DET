"""Render M013 using the actual native LVGL page and transforms."""
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sources=sorted((ROOT/'managed_components/lvgl__lvgl/src').rglob('*.c'))
sources+=sorted((ROOT/'main/ui').rglob('*.c'))+[ROOT/'tests/ui_preview/category_main.c']
args=['-std=c11','-O1','-UNDEBUG','-DLV_CONF_INCLUDE_SIMPLE','-DLV_LVGL_H_INCLUDE_SIMPLE']
args+=['-I'+str(ROOT/p) for p in ['tests/ui_preview','managed_components/lvgl__lvgl','main/ui']]
args+=[str(p) for p in sources]+['-o',str(ROOT/'tmp/category_preview.exe')]
rsp=ROOT/'tmp/category_preview.rsp';rsp.write_text(subprocess.list2cmdline(args))
subprocess.run([str(ROOT/'tmp/host-tools/ziglang/zig.exe'),'cc','@'+str(rsp)],check=True)
frames=ROOT/'tmp/category-frames';frames.mkdir(exist_ok=True)
subprocess.run([str(ROOT/'tmp/category_preview.exe'),str(frames)],check=True)
out=ROOT/'docs/ui';out.mkdir(parents=True,exist_ok=True)
images=[Image.open(frames/f'{i:03d}.ppm').convert('RGB') for i in range(307)]
for frame in [0,60,150,240,306]:
    for box in [(224,3,280,32),(467,3,523,32),(467,61,605,93),(32,159,107,195)]:
        region=images[frame].crop(box)
        assert sum(max(region.getpixel((x,y)))>50 for y in range(region.height) for x in range(region.width))>150, (frame,box,'missing text')
images[0].save(out/'m013-first.png');images[-1].save(out/'m013-final.png')
# GIF uses 10ms units; alternate 10/20/20 for the source 60fps average.
images[0].save(out/'m013-native.gif',save_all=True,append_images=images[1:],
               duration=([10,20,20]*103)[:307],loop=0,disposal=2)
sheet=Image.new('RGB',(1296,600))
for i,f in enumerate([0,26,60,150,240,306]):sheet.paste(images[f],((i%2)*648,(i//2)*200))
sheet.save(out/'m013-states.png')
print('Native M013 previews: '+str(out))
