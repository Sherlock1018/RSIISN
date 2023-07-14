# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import time
import torch
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
img = "/home/zwc/Documents/mmdetection/data/coco/test.png"
config = '/home/zwc/Documents/mmdetection/configs/swin/cascade_mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py'
checkpoint = '/home/zwc/Documents/mmdetection/work_dirs/cascade-swin-ori-without-traintrick/epoch_12.pth'
model = init_detector(config, checkpoint, device='cuda:0')
time_start = time.time()
for i in range(100):
    result = inference_detector(model, img)
time_end = time.time()
averge_fps = 1/((time_end - time_start)/100)
print("time is:", averge_fps)
 



