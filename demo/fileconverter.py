import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy
import json
import png
import numpy as np
import os

def convert(img_path, json_file, mode, aa, bb):
    print(mode)
    Js = {
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "wall", "id": 1, "name": "wall"},
            {"supercategory": "chair", "id": 2, "name": "chair"},
            {"supercategory": "floor", "id": 3, "name": "floor"},
            {"supercategory": "table", "id": 4, "name": "table"},
            {"supercategory": "door", "id": 5, "name": "door"},
            {"supercategory": "cabinet", "id": 7, "name": "cabinet"},
            {"supercategory": "shelf", "id": 8, "name": "shelf"},
            {"supercategory": "desk", "id": 9, "name": "desk"},
            {"supercategory": "bed", "id": 11, "name": "bed"},
            {"supercategory": "toilet", "id": 17, "name": "toilet"},
            {"supercategory": "curtain", "id": 21, "name": "curtain"},
            {"supercategory": "refrigerator", "id": 27, "name": "refrigerator"},
            {"supercategory": "sofa", "id": 13, "name": "sofa"},
            {"supercategory": "sink", "id": 14, "name": "sink"},
            {"supercategory": "shower curtain", "id": 55, "name": "shower curtain"},
            {"supercategory": "window", "id": 16, "name": "window"},
            {"supercategory": "picture", "id": 15, "name": "picture"},
            {"supercategory": "bookshelf", "id": 18, "name": "bookshelf"},
            {"supercategory": "counter", "id": 35, "name": "counter"},
        ]
    }
    img_id = 0
    anno_id = 0
    f = open(img_path, "r")
    for ii in range(aa):
        aline = f.readline()
        for jj in range(0, 5000, bb):
            if not os.path.isfile('/home/wenjing/storage/ScanNetv2/' + mode + '_label/' + aline[:12] + '/label-filt/' + str(jj) +'.png'):
                break
            label = Image.open('/home/wenjing/storage/ScanNetv2/' + mode + '_label/' + aline[:12] + '/label-filt/' + str(jj) +'.png')
            label = np.array(label)
            # if label is None:
            #     break
            print(aline[:12] + '/' + str(jj) +'.png')
            # hh, ww = label.shape[0], label.shape[1]
            Js['images'].append(
                {"file_name":  aline[:12] + '/color/' + str(jj) + ".jpg", "height": 968, "width": 1296,
                 "id": img_id})
            bbox = {}
            segment = {}
            class_20=np.array([1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 21, 27, 35, 55], dtype=np.int)
            classes_full = np.unique(label)
            class_label = []
            for i in classes_full:
                if i in class_20:
                    class_label.append(i)
            for cl in class_label:
                    cl2 = np.array(cl)
                    binary = cv.compare(label, cl2, cmpop=cv.CMP_EQ)
                    area = np.count_nonzero(binary)
                    _, contour, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    maxlen = 0
                    for i in range(len(contour)):
                        if len(contour[i]) > maxlen:
                            maxlen_ind = i
                            maxlen = len(contour[i])
                    d = contour[maxlen_ind]
                    d = np.squeeze(d)
                    a = np.min(d, axis=0)
                    b = np.amax(d, axis=0)
                    w = b[0] - a[0]
                    h = b[1] - a[1]
                    if w>0 and h>0:
                        bbox[cl] = [a[0].item(), a[1].item(), w.item(), h.item()]
                        d = np.reshape(d, (1, -1))
                        d = d.tolist()
                        segment[cl] = d

                        Js['annotations'].append(
                            {'segmentation': segment[cl], 'area': area, 'iscrowd': 0, 'image_id': img_id,
                             'bbox': bbox[cl], 'category_id': cl.item(), 'id': anno_id, 'depth': mode + '_depth/' + aline[:12]+ '/depth/' + str(jj) + '.png'})
                        # Js['annotations'].append(
                        #     {'segmentation': segment[cl], 'area': area, 'iscrowd': 0, 'image_id': img_id,
                        #      'bbox': bbox[cl], 'category_id': cl.item(), 'id': anno_id, 'depth': segment[cl]})
                        anno_id += 1
            img_id += 1
    print(mode, "data number :", img_id)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(Js, indent=4)
    json_fp.write(json_str)
    json_fp.close()

if __name__ == '__main__':
    img_path = '/home/wenjing/storage/ScanNetv2/train.txt'
    json_file = '/home/wenjing/anno/train_depth_real.txt'
    convert(img_path, json_file, mode = 'train', aa = 100, bb = 100)
    # a in range 1,119
    img_path = '/home/wenjing/storage/ScanNetv2/val.txt'
    json_file = '/home/wenjing/anno/val_depth_real.txt'
    convert(img_path, json_file, mode='val', aa=50, bb= 100)

    # b in range 1, 45
