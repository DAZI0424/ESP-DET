"""Measure vector dot keyframes; no video pixels are embedded in firmware."""
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
cap = cv2.VideoCapture(str(ROOT / '886ee293382508b063daf651dbeac3d8.mp4'))
frames = []
for frame in range(19, 49):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = cap.read()
    assert ok
    gray = cv2.cvtColor(bgr[165:200,48:87], cv2.COLOR_BGR2GRAY)
    count, labels, stats, centers = cv2.connectedComponentsWithStats((gray > 65).astype('uint8'))
    dots = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 3: continue
        ys, xs = np.where(labels == label)
        weights = gray[ys,xs].astype(float)
        cx, cy = np.average(xs,weights=weights), np.average(ys,weights=weights)
        opacity = int(np.percentile(weights,90))
        if area > 33:
            # Resolve touching dots into two overlapping native circles.
            points = np.array([xs-cx,ys-cy])
            eigenvalues, vectors = np.linalg.eigh(np.cov(points,aweights=weights))
            direction = vectors[:,-1]
            separation = max(1.0, float(np.sqrt(max(0,eigenvalues[-1]-eigenvalues[0]))))
            for sign in [-1,1]:
                dots.append((cx+sign*direction[0]*separation,cy+sign*direction[1]*separation,5,opacity))
        else:
            diameter = min(6,max(3,round((4*area/np.pi)**0.5)))
            dots.append((cx,cy,diameter,opacity))
    assert len(dots) <= 8, (frame,dots)
    # Store rectangle coordinates, diameter and opacity only.
    row = [(round(cx-(size-1)/2),round(cy-(size-1)/2),size,opa) for cx,cy,size,opa in dots]
    row += [(0,0,0,0)]*(8-len(row))
    frames.append(row)
cap.release()
text = '// Measured vector motion from source frames 19..48 at 30 fps. No image data.\n'
text += '#pragma once\n#include <stdint.h>\ntypedef struct { uint8_t x, y, diameter, opacity; } ui_loading_dot_key_t;\n'
text += 'static const ui_loading_dot_key_t loading_keys[30][8] = {\n'
for i,row in enumerate(frames):
    text += '    {' + ', '.join('{'+','.join(map(str,dot))+'}' for dot in row) + '}, // source frame '+str(i+19)+'\n'
text += '};\n'
(ROOT/'main/ui/ui_loading_motion.h').write_text(text)
print('Generated 30 vector keyframes, up to 8 native circles, 960 bytes.')
