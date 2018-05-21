import numpy as np
import sys, pylab
import operator
import csv, scipy
import pickle

scale = [1.0, 0.8, 0.6]
asp_ratio = [4.0/3, 3/4.0, 16.0/9]
def getSubImg(I_W, I_H, scale, aspect_ratio, stride_ratio):
    scale_x = 0
    scale_y = 0
    asp_orig = I_W/float(I_H)
    if aspect_ratio > asp_orig:
        W = I_W*scale
        H = W/aspect_ratio
    else:
        H = scale*I_H
        W = H*aspect_ratio

    stride = stride_ratio*min(I_W, I_H)
    if I_W-W > 0:
        step_x = (I_W-W) / np.ceil((I_W - W)/stride)
    else:
        step_x = 0
    if I_H-H > 0:
        step_y = (I_H-H) / np.ceil((I_H - H)/stride)
    else:
        step_y = 0

    crops = []
    for i in range(int(np.ceil((I_W - W)/stride)+1)):
        for j in range(int(np.ceil((I_H - H)/stride)+1)):
            crops.append([i*step_x/I_W, j*step_y/I_H, min((i*step_x+W)/I_W,1.0), min((j*step_y+H)/I_H,1.0)])

    return crops

def iou(a, b):
  x1 = max(a[0], b[0])
  y1 = max(a[1], b[1])
  x2 = min(a[2], b[2])
  y2 = min(a[3], b[3])
  ar1 = (a[2]-a[0])*(a[3]-a[1])
  ar2 = (b[2]-b[0])*(b[3]-b[1])
  ov = (x2-x1)*(y2-y1)
  if x2 > x1 and y2 > y1:
    return float(ov)/(ar1+ar2-ov)
  else:
    return 0

def doNMS(crops, thresh):
  res = []
  idx = []
  count = 0
  res.append(crops.pop(0))
  idx.append(count)

  while crops:
    count += 1
    r = crops.pop(0)
    flag = True
    for kk in res:
      if iou(kk, r) > thresh:
        flag = False
        break
    if flag:
      res.append(r)
      idx.append(count)

  return res, idx


anchor_crops = []
asp_idx = 0

I_W = 800
I_H = 600
for aa in asp_ratio:
    for ss in scale:
        tmp = getSubImg(I_W, I_H, ss, aa, 0.1)
        [r.append(asp_idx) for r in tmp]
        anchor_crops.extend(tmp)
    asp_idx += 1


I_W = 600
I_H = 800
for aa in asp_ratio:
    for ss in scale:
        tmp = getSubImg(I_W, I_H, ss, aa, 0.1)
        [r.append(asp_idx) for r in tmp]
        anchor_crops.extend(tmp)
    asp_idx += 1

I_W = 500
I_H = 400
for aa in asp_ratio:
    for ss in scale:
        tmp = getSubImg(I_W, I_H, ss, aa, 0.1)
        [r.append(asp_idx) for r in tmp]
        anchor_crops.extend(tmp)
    asp_idx += 1

I_W = 400
I_H = 500
for aa in asp_ratio:
    for ss in scale:
        tmp = getSubImg(I_W, I_H, ss, aa, 0.1)
        [r.append(asp_idx) for r in tmp]
        anchor_crops.extend(tmp)
    asp_idx += 1

I_W = 500
I_H = 500
for aa in asp_ratio:
    for ss in scale:
        tmp = getSubImg(I_W, I_H, ss, aa, 0.1)
        [r.append(asp_idx) for r in tmp]
        anchor_crops.extend(tmp)
    asp_idx += 1

I_W = 1600
I_H = 900
for aa in asp_ratio:
    for ss in scale:
        tmp = getSubImg(I_W, I_H, ss, aa, 0.1)
        [r.append(asp_idx) for r in tmp]
        anchor_crops.extend(tmp)
    asp_idx += 1

I_W = 900
I_H = 1600
for aa in asp_ratio:
    for ss in scale:
        tmp = getSubImg(I_W, I_H, ss, aa, 0.1)
        [r.append(asp_idx) for r in tmp]
        anchor_crops.extend(tmp)
    asp_idx += 1


anchor_crops, _ = doNMS(anchor_crops, 0.85)

print len(anchor_crops)
with open('anchor_crops_dense.pkl', 'wb') as f:
    pickle.dump(anchor_crops, f)