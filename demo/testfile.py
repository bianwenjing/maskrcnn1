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

writer = SummaryWriter()
f = open('/home/wenjing/storage/ScanNetv2/test.txt', "r")
for ii in range(10):
    aline = f.readline()
    for jj in range(0, 1, 100):
        if not os.path.isfile('/home/wenjing/storage/ScanNetv2/test/' + aline[:12] + '/color/' + str(jj) + '.jpg'):
            break
        image = cv2.imread('/home/wenjing/storage/ScanNetv2/test/' + aline[:12] + '/color/' + str(jj) + '.jpg')
        predictions = coco_demo.run_on_opencv_image(image)
        # predictions = torch.from_numpy(predictions)
        writer.add_image(str(ii), predictions,1, dataformats = 'HWC')
writer.close()