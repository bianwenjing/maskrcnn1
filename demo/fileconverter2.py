import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy
import json
import png
import numpy as np
import os
import csv
from pycocotools import mask as cocomask

def convert(img_path, json_file, mode, aa, bb):
    print(mode)
    Js = {
        "images": [],
        "annotations": [],
        "categories": []
    }
#######################################################################################################################
    K = {}
    with open("/home/wenjing/scannetv2-labels.combined.tsv") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        lines = []
        for line in tsvreader:
            lines.append(line)
        for i in range(1, len(lines)):
            # raw_category = lines[i][1]
            # category = lines[i][2]
            id = lines[i][0]
            K[id] = 0
    with open('/home/wenjing/storage/anno/train_git_many_100.txt') as json_f:
        data = json.load(json_f)

    for i in data['annotations']:
        category_id = str(i['category_id'])
        K[category_id] += 1

    valid_category = []
    for key, item in K.items():
        if item > 100:
            valid_category.append(key)
    json_f.close()
########################################################################################################################
    # with open("/home/wenjing/scannetv2-labels.combined.tsv") as tsvfile:
    #     tsvreader = csv.reader(tsvfile, delimiter="\t")
    #     lines = []
    #     for line in tsvreader:
    #         lines.append(line)
    for i in range(1, len(lines)):
        raw_category = lines[i][1]
        category = lines[i][2]
        id = lines[i][0]
        if id in valid_category:
            Js["categories"].append({"supercategory": category, "id": int(id), "name": category})
#########################################################################################################
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

            # intrinsic = []
            # intr_path = '/home/wenjing/storage/ScanNetv2/' + mode + '_intrinsics/' + aline[:12] +'/intrinsic/intrinsic_depth.txt'
            # intr_file = open(intr_path, "r")
            # matrix =intr_file.readlines()
            # f1 = matrix[0].split()[0]
            # intrinsic.append(float(f1))
            # f2 = matrix[1].split()[1]
            # intrinsic.append(float(f2))
            # intr_file.close()

            # bbox = {}
            # segment = {}
            # class_20=np.array([1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 21, 27, 35, 55], dtype=np.int)
            classes_full = np.unique(label)
            if classes_full[0] == 0:
                classes_full = np.delete(classes_full, 0)
            class_label = []
            for i in classes_full:
                if i in valid_category:
                    class_label.append(i)
            for cl in class_label:
                    cl2 = np.array(cl)
                    binary = cv.compare(label, cl2, cmpop=cv.CMP_EQ)
                    # area = np.count_nonzero(binary)
                    _, contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    ######################################################################################
                    segmentation = []
                    for contour in contours:
                        if contour.size >= 6:
                            segmentation.append(contour.flatten().tolist())
                    if len(segmentation) > 0:
                        RLEs = cocomask.frPyObjects(segmentation, label.shape[0], label.shape[1])
                        RLE = cocomask.merge(RLEs)
                        area = cocomask.area(RLE)
                        [x, y, w, h] = cv.boundingRect(binary)

                        if w > 0 and h > 0:
                            Js['annotations'].append(
                                {
                                 'segmentation': segmentation,
                                 'area': area.item(),
                                 'iscrowd': 0,
                                 'image_id': img_id,
                                 'bbox': [x, y, w, h],
                                 'category_id': cl.item(),
                                 'id': anno_id,
                                 'depth': mode + '_depth/' + aline[:12] + '/depth/' + str(jj) + '.png'})
                            anno_id += 1
                    #########################################################################################
                    # contour_sizes = []
                    # valid_contour_id = []
                    # for contour in contours:
                    #     # i (num, 1, 2)
                    #     contour_sizes.append(contour.size)
                    # max_contour = max(contour_sizes)
                    # for i, contour_size in enumerate(contour_sizes):
                    #     ratio = contour_size/max_contour
                    #     if ratio > 0.2 and contour_size > 6:
                    #         valid_contour_id.append(i)
                    # if len(valid_contour_id) == 1:  # polygon
                    #     d = contours[valid_contour_id[0]]
                    #     d = np.squeeze(d)
                    #     if len(d.shape) == 2:
                    #         a = np.min(d, axis=0)
                    #         b = np.amax(d, axis=0)
                    #         w = b[0] - a[0]
                    #         h = b[1] - a[1]
                    #         if w > 0 and h > 0:
                    #             bbox[cl] = [a[0].item(), a[1].item(), w.item(), h.item()]
                    #             d = np.reshape(d, (1, -1))
                    #             d = d.tolist()
                    #             segment[cl] = d
                    # else: # RLE
                    #     segmentation = []
                    #     for i in valid_contour_id:
                    #
                    #     RLEs = cocomask.frPyObjects()

                    ############################################################################################
                    # maxlen = 0
                    # for i in range(len(contour)):
                    #     if len(contour[i]) > maxlen:
                    #         maxlen_ind = i
                    #         maxlen = len(contour[i])
                    # d = contour[maxlen_ind]
                    # d = np.squeeze(d)
                    # if len(d.shape) == 2 and d.shape[0]>=3:
                    #     a = np.min(d, axis=0)
                    #     b = np.amax(d, axis=0)
                    #     w = b[0] - a[0]
                    #     h = b[1] - a[1]
                    #     if w > 0 and h > 0:
                    #         bbox[cl] = [a[0].item(), a[1].item(), w.item(), h.item()]
                    #         d = np.reshape(d, (1, -1))
                    #         d = d.tolist()
                    #         segment[cl] = d

                    ######################################################################################################
                            # Js['annotations'].append(
                            #     {'intrinsic': intrinsic, 'segmentation': segment[cl], 'area': area, 'iscrowd': 0, 'image_id': img_id,
                            #      'bbox': bbox[cl], 'category_id': cl.item(), 'id': anno_id, 'depth': mode + '_depth/' + aline[:12]+ '/depth/' + str(jj) + '.png'})
                            # Js['annotations'].append(
                            #     {'segmentation': segment[cl], 'area': area, 'iscrowd': 0,
                            #      'image_id': img_id,
                            #      'bbox': bbox[cl], 'category_id': cl.item(), 'id': anno_id,
                            #      'depth': mode + '_depth/' + aline[:12] + '/depth/' + str(jj) + '.png'})
                            # anno_id += 1
            img_id += 1
    print(mode, "data number :", img_id)
    print('category_number: ', len(valid_category))
    # json_fp = open(json_file, 'w')
    # json_str = json.dumps(Js, indent=4)
    # json_fp.write(json_str)
    with open(json_file, 'w') as json_fp:
        json.dump(Js, json_fp, indent=4)
    json_fp.close()
    f.close()
    tsvfile.close()

if __name__ == '__main__':
    img_path = '/home/wenjing/storage/ScanNetv2/scannetv2_train.txt'
    json_file = '/home/wenjing/storage/anno/train_git_many_100_reduced.txt'
    convert(img_path, json_file, mode='train_scan', aa=1201, bb=100)
    # a in range 1,1201
    img_path = '/home/wenjing/storage/ScanNetv2/scannetv2_val.txt'
    json_file = '/home/wenjing/storage/anno/val_git_many_100_reduced.txt'
    convert(img_path, json_file, mode='val_scan', aa=312, bb=100)
    # img_path = '/home/wenjing/storage/ScanNetv2/scannetv2_train.txt'
    # json_file = '/home/wenjing/storage/anno/ground_train.txt'
    # convert(img_path, json_file, mode='train_scan', aa=10, bb=5000)
    # b in range 1, 45
