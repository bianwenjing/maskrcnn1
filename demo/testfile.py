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
from maskrcnn_benchmark.data.datasets.mydataset import ScanNetDataset

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

# config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
config_file = "../configs/caffe2/channel_r50_c4.yaml"

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


OBJECT_DEPTH = False
WHOLE_DEPTH = False
SEG_GROUND = True

if SEG_GROUND:
    anno = '/home/wenjing/storage/anno/ground_train_resize.txt'
    root = '/home/wenjing/storage/ScanNetv2/val_scan'
    a = ScanNetDataset(anno, root, True)

writer = SummaryWriter()
f = open('/home/wenjing/storage/ScanNetv2/scannetv2_val.txt', "r")
for ii in range(10):
    aline = f.readline()
    for jj in range(0, 1, 100):
        if not os.path.isfile('/home/wenjing/storage/ScanNetv2/val_scan/' + aline[:12] + '/color/' + str(jj) + '.jpg'):
            break
        image = Image.open('/home/wenjing/storage/ScanNetv2/val_scan/' + aline[:12] + '/color/' + str(jj) + '.jpg')
        image = image.resize((320, 240))
        image = np.array(image)
        print('@@@@@@@@@@@@@@@@@', aline[:12])
        predictions = coco_demo.run_on_opencv_image(image)
        writer.add_image(str(ii), predictions, 0, dataformats='HWC')
        ################################################################################################
        if SEG_GROUND:
            img, target, idx = a[ii]
            target_tensor = target.get_field('masks').get_mask_tensor()
            # print('""""""""""""', target_tensor.shape)
            if len(target_tensor.shape) == 2:
                target_tensor = target_tensor[None, :, :]
            target_tensor = target_tensor[:, None, :, :]
            target.add_field('mask', target_tensor)
            ground_truth = coco_demo.run_on_ground_truth(image, target)
            writer.add_image(str(ii) + 'ground_truth', ground_truth, 0, dataformats='HWC')
        ##########################################################################################################
        ##################################object depth prediction################################################
        if OBJECT_DEPTH:
            depth_pred = coco_demo.run_on_opencv_image(image, depth='object')
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

        ####################################################################################################
        ###################################depth ground truth###########################################
        if OBJECT_DEPTH or WHOLE_DEPTH:
            depth_target = Image.open(
                '/home/wenjing/storage/ScanNetv2/val_scan_depth/' + aline[:12] + '/depth/' + str(jj) + '.png')
            depth_target = depth_target.resize((320, 240))
            depth_target = np.array(depth_target)
        if OBJECT_DEPTH:
            d_min = min(np.min(depth_pred), np.min(depth_target))
            d_max = max(np.max(depth_pred), np.max(depth_target))
            depth_target_scaled = colored_depthmap(depth_target, d_min, d_max)
            depth_pred_scaled = colored_depthmap(depth_pred, d_min, d_max)
            save_path_pred = '/home/wenjing/test1/' + str(ii) + '_pred.png'
            plt.imsave(save_path_pred, depth_pred_scaled)
            depth_pred_scaled = Image.open(save_path_pred)
            depth_pred_scaled = np.array(depth_pred_scaled)

            save_path_target = '/home/wenjing/test1/' + str(ii) + '_target.png'
            plt.imsave(save_path_target, depth_target_scaled)
            depth_target_scaled = Image.open(save_path_target)
            depth_target_scaled = np.array(depth_target_scaled)
            writer.add_image(str(ii) + 'depth_ground', depth_target_scaled, 0, dataformats='HWC')
            writer.add_image(str(ii) + '_depth', depth_pred_scaled, 0, dataformats='HWC')

        ###############################################################################################
        #################################whole depth##################################################
        if WHOLE_DEPTH:
            whole_depth_pred = coco_demo.run_on_opencv_image(image, depth='whole')
            whole_depth_pred = whole_depth_pred[0]
            # print('£££££££££££', whole_depth_pred.shape)
            whole_depth_pred[whole_depth_pred < 0] = 0
            whole_depth_pred = np.uint32(whole_depth_pred)
        ####################################################################################################
        ###################################depth ground truth2 ###########################################
            d_min = min(np.min(whole_depth_pred), np.min(depth_target))
            d_max = max(np.max(whole_depth_pred), np.max(depth_target))
            depth_target_scaled = colored_depthmap(depth_target, d_min, d_max)
            whole_depth_pred_scaled = colored_depthmap(whole_depth_pred, d_min, d_max)
            save_path_pred = '/home/wenjing/test1/' + str(ii) + '_pred_whole.png'
            plt.imsave(save_path_pred, whole_depth_pred_scaled)
            depth_pred_scaled = Image.open(save_path_pred)
            depth_pred_scaled = np.array(depth_pred_scaled)

            save_path_target = '/home/wenjing/test1/' + str(ii) + '_target_whole.png'
            plt.imsave(save_path_target, depth_target_scaled)
            depth_target_scaled = Image.open(save_path_target)
            depth_target_scaled = np.array(depth_target_scaled)
            writer.add_image(str(ii) + 'whole_depth_ground', depth_target_scaled, 0, dataformats='HWC')
            writer.add_image(str(ii) + 'whole_depth', depth_pred_scaled, 0, dataformats='HWC')

writer.close()
f.close()


