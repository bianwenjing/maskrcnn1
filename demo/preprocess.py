from PIL import Image
import numpy as np
# from scipy.sparse import csr_matrix
from PIL import ImageFilter
import matplotlib.pyplot as plt
import os
from maskrcnn_benchmark.utils.miscellaneous import mkdir
def preprocess_depth_map(img):
    b = 9
    img = np.array(img)
    img_cut = img[b:-b, b:-b]
    img_cut = Image.fromarray(img_cut)

    img_filtered = np.array(img_cut.filter(ImageFilter.MedianFilter(size=17)))
    mask = img_cut == 0
    result1 = img_cut * np.invert(mask) + img_filtered * mask

    mask = result1 == 0
    still_blank = np.any(mask)
    i = 0
    while still_blank and i < 100:
        result1 = Image.fromarray(result1)
        img_filtered = np.array(result1.filter(ImageFilter.MedianFilter(size=11)))
        result1 = result1 * np.invert(mask) + img_filtered * mask

        i += 1
        mask = result1 == 0
        still_blank = np.any(mask)
    img[b:-b, b:-b] = result1
    return img

def fill_depth_colorization(imgRgb, imgDepth, alpha=1):
    # imgRgb PIL Image

    # imgIsNoise = (imgDepth == 0)
    maxImgAbsDepth = max(imgDepth)
    imgDepth = imgDepth/maxImgAbsDepth

    H = imgDepth.shape[0]
    W = imgDepth.shape[1]
    numPix = H * W
    indsM = np.arange(0, numPix).reshape(H, W)

    knownValMask = (imgDepth != 0)

    grayImg = imgRgb.convert('L')

    winRad = 1
    len = 0
    absImgNdx = -1

    cols = np.zeros((numPix * (2*winRad+1) ^ 2, 1))
    rows = np.zeros((numPix * (2*winRad+1) ^ 2, 1))
    vals = np.zeros((numPix * (2*winRad+1) ^ 2, 1))
    gvals = zeros((1, (2*winRad+1) ^ 2))

    for j in range(W):
        for i in range(H):
            absImgNdx = absImgNdx + 1
            nWin = 0  # Counts the number of points in the current window
            for ii in range(max(1, i-winRad)-1,min(i+winRad, H)):
                for jj in range(max(1, j-winRad)-1, min(j+winRad, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len] = absImgNdx
                    cols[len] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]
                    len = len + 1
                    nWin = nWin + 1
            curVal = grayImg[i,j]
            gvals[nWin] = curVal
            c_var = np.mean(gvals[0:nWin+1]-np.mean(gvals[0:nWin+1])^2)
            csig = c_var * 0.6
            mgv = min((gvals[0:nWin]-curVal)^2)

            if csig < (-mgv/np.log(0.01)):
                csig = -mgv / np.log(0.01)
            if csig < 0.000002:
                csig = 0.000002
            gvals[0:nWin] = np.exp(-(gvals[0:nWin]-curVal)^2/csig)
            gvals[0:nWin] = gvals[0:nWin]/np.sum(gvals[0:nWin])
            vals[len-nWin:len] = -gval[0:nWin]

    vals = vals[0:len]
    cols = cols[0:len]
    rows = rows[0:len]
    A = csr_matrix((vals, (rows, cols)), shape = (numPix, numPix)).toarray()

    rows = np.arange(0, np.size(knownValMask))
    cols = np.arange(0, np.size(knownValMask))
    vals = knownValMask.reshape(-1) * alpha
    G = csr_matrix((vals, (rows, cols)), shape = (numPix, numPix)).toarray()

    new_vals = (vals * imgDepth.reshape(-1)) / (A + G)
    new_vals = new_vals.reshape((H,W))
    denoisedDepthImg = new_vals * maxImgAbsDepth
    return denoisedDepthImg

if __name__ == '__main__':
    img_path = '/home/wenjing/storage/ScanNetv2/scannetv2_val.txt'
    f = open(img_path, "r")
    mode = 'val'
    for ii in range(312):
        aline = f.readline()
        for jj in range(0, 20000, 30):
            img_dir = '/home/wenjing/storage/ScanNetv2/' + mode + '_scan/' + aline[:12] + '/color/' + str(jj) +'.jpg'
            if not os.path.isfile(img_dir):
                break
            # imgRgb = Image.open(img_dir).resize((320, 240))
            depth_dir = '/home/wenjing/storage/ScanNetv2/' + mode + '_scan_depth/' + aline[:12] + '/depth/' + str(jj) +'.png'
            print(aline[:12] + '/' + str(jj) + '.png')
            imgDepth = Image.open(depth_dir).resize((320,240))
            # imgDepth = np.array(imgDepth)
            # denoisedDepthImg = fill_depth_colorization(imgRgb, imgDepth, alpha = 1)
            img_dep = preprocess_depth_map(imgDepth)
            save_path = '/home/wenjing/storage/ScanNetv2/preprocess/' + mode + '_scan_depth/' + aline[:12] + '/depth/'
            mkdir(save_path)
            # plt.imsave(save_path + str(jj) +'.png', img_dep)
            Image.fromarray(img_dep.astype(np.int16)).save(save_path + str(jj) +'.png')
    f.close()
