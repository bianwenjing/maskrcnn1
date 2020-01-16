import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from torch.utils.tensorboard import SummaryWriter
import os
import torch

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False,
)
def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
#     return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C
    return 255*depth_relative

writer = SummaryWriter()
f = open('/home/wenjing/storage/ScanNetv2/test.txt', "r")
for ii in range(10):
    aline = f.readline()
    for jj in range(0, 1, 100):
        if not os.path.isfile('/home/wenjing/storage/ScanNetv2/test/' + aline[:12] + '/color/' + str(jj) + '.jpg'):
            break
        image = cv2.imread('/home/wenjing/storage/ScanNetv2/test/' + aline[:12] + '/color/' + str(jj) + '.jpg')
        print('@@@@@@@@@@@@@@@@@', aline[:12])
        predictions = coco_demo.run_on_opencv_image(image)
        writer.add_image(str(ii), predictions, 0, dataformats='HWC')

        depth_pred = coco_demo.run_on_opencv_image(image, depth=True)
        depth_pred = depth_pred[0]
        depth_pred[depth_pred < 0] = 0
        depth_pred = np.uint32(depth_pred)
        # np.save('/home/wenjing/dep.npy', depth_pred)
        # predictions = torch.from_numpy(predictions)
        # depth_pred = depth_pred[:, :, [1, 2, 0]]
        # depth_pred[depth_pred<0] = 0
        # depth_pred = np.uint8(depth_pred)
        # print(np.unique(depth_pred))
        # depth_pred = Image.fromarray(depth_pred)
        # depth_pred.save("/home/wenjing/image.png", "PNG")

        depth_target = Image.open('/home/wenjing/storage/ScanNetv2/test_depth/' + aline[:12] + '/depth/' + str(jj) + '.png')
        depth_target = depth_target.resize((1296, 968))
        depth_target = np.array(depth_target)
        d_min = min(np.min(depth_pred), np.min(depth_target))
        d_max = max(np.max(depth_pred), np.max(depth_target))
        depth_target_scaled = colored_depthmap(depth_target, d_min, d_max)
        depth_pred_scaled = colored_depthmap(depth_pred, d_min, d_max)
        writer.add_image(str(ii) + '_depth', depth_pred, 0, dataformats='HW')
        writer.add_image(str(ii)+'depth_ground', depth_target, 0, dataformats = 'HW')
writer.close()
f.close()


