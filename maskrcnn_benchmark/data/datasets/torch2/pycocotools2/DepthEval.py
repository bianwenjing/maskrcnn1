from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import time
import numpy as np
import copy
import pycocotools.mask as maskUtils
from PIL import Image
import math

class DEPTHeval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        # super(DEPTHeval,self).__init__(cocoGt, cocoDt, iouType)

        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        ###################################################################
        self.depth_error = {}
        ####################################################################
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self):
        '''
                Prepare ._gts and ._dts for evaluation based on params
                :return: None
                '''

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']

        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation


        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
            # print('££££££££££££££££££', gt['image_id'], gt['category_id'])
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
          # number of ground truth items (anno id+1)
        # len(self._dts): 111, number of detected items
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        if p.maxDets:
            p.maxDets = sorted(p.maxDets)
        self.params = p
        self._prepare()


        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]



        # if p.iouType == 'depth':
        #     compute_depth_metrics = self.compute_depth_metrics
        #     self.depth_error = {(imgId, catIds): compute_depth_metrics(imgId, catId) \
        #                         for imgId in p.imgIds
        #                         for catId in catIds}
        #     print('£££££££££££££££££££££££££', self.depth_error)
        # else:
        #     if p.iouType == 'segm' or p.iouType == 'bbox':
        #         computeIoU = self.computeIoU
        #
        #     elif p.iouType == 'keypoints':
        #         computeIoU = self.computeOks
        #
        #     self.ious = {(imgId, catId): computeIoU(imgId, catId) \
        #                  for imgId in p.imgIds
        #                  for catId in catIds}
        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType =='depth':
            computeIoU = self.compute_depth_metrics
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        if p.iouType == 'whole_depth':
            computeIoU = self.compute_whole_depth_metrics


        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                    for imgId in p.imgIds
                    for catId in catIds}
        if p.iouType == 'depth' or p.iouType == 'whole_depth':
            error = []
            for imgId in p.imgIds:
                for catId in catIds:
                    x = computeIoU(imgId, catId)
                    if x != []:
                        error.append(x)
            error = np.asarray(error)
            self.mean_error = np.mean(error, axis=0)
            # print('##################', self.mean_error)
        else:
            evaluateImg = self.evaluateImg
            maxDet = p.maxDets[-1]
            self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                             for catId in catIds
                             for areaRng in p.areaRng
                             for imgId in p.imgIds
                             ]
            self._paramsEval = copy.deepcopy(self.params)

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))
    def compute_whole_depth_metrics(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]

        if len(gt) == 0 or len(dt) == 0:
            return []
        # inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        # dt = [dt[i] for i in inds]
        # if len(dt) > p.maxDets[-1]:
        #     dt = dt[0:p.maxDets[-1]]

        g = [g['depth'] for g in gt]
        d = [d['whole_depth'] for d in dt]


        depth_d = Image.open(d[0])
        width, height = depth_d.size
        depth_d = np.array(depth_d)
        depth_g = Image.open('/home/wenjing/storage/ScanNetv2/' + g[0]).resize((width, height))
        depth_g = np.array(depth_g)

        # remove zeros to avoid divide by zero
        mask1 = depth_d != 0
        mask2 = depth_g != 0
        mask = mask1*mask2
        depth_g = depth_g[mask]
        depth_d = depth_d[mask]

        thresh = np.maximum((depth_g / depth_d), (depth_d / depth_g))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_diff = np.abs(depth_g - depth_d)

        mse = np.mean(np.power(abs_diff,2))
        rmse = np.sqrt(mse)

        # print('£££££££££££££££££££££', np.unique(depth_d))
        mse_log = (np.log(depth_g) - np.log(depth_d)) ** 2
        rmse_log = np.sqrt(mse_log.mean())

        abs_rel = np.mean(np.abs(depth_d - depth_g) / depth_g)
        sq_rel = np.mean(((depth_g - depth_d) ** 2) / depth_g)


        log10_error = np.abs(np.log10(depth_g) - np.log10(depth_d))
        log10_mean = np.mean(log10_error)
        # metrics = [abs_rel, sq_rel, rmse, rmse_log, log10_mean, a1, a2, a3]

        inv_output = 1 / depth_d
        inv_target = 1 / depth_g
        abs_inv_diff = (inv_output - inv_target).abs()
        imae = float(abs_inv_diff.mean())
        irmse = math.sqrt((np.power(abs_inv_diff, 2)).mean())
        mae = np.mean(abs_diff)
        log_mae = np.mean(np.abs(np.log(depth_g) - np.log(depth_d)))
        scale_invar =

        metrics = [abs_rel, imae, irmse, log_mae, rmse_log, mae, rmse, scale_invar, sq_rel]
        return metrics

    def compute_depth_metrics(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]

        if len(gt) == 0 or len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        # if len(dt) > p.maxDets[-1]:
        #     dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'depth':
            g = [g['depth'] for g in gt]
            d = [d['depth'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')
        depth_g_ = Image.open('/home/wenjing/storage/ScanNetv2/' + g[0]).resize((1296, 968))
        depth_g_ = np.array(depth_g_)
        # depth_g.shape (968,1296)

        depth_d = np.zeros((968,1296))
        for d_part in d:
            mask = depth_d==0
            depth_i = Image.open(d_part)
            depth_i = np.array(depth_i)
            # depth_d = np.maximum(depth_d, depth_i)
            depth_d += depth_i * mask

        depth_g_[(depth_d==0)]=0

        #remove zeros to avoid divide by zero
        depth_g = depth_g_[depth_g_ != 0]
        depth_d = depth_d[depth_g_ != 0]

        thresh = np.maximum((depth_g/depth_d), (depth_d/depth_g))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (depth_d - depth_g)**2
        rmse = np.sqrt(rmse.mean())
        # print('@@@@@@@@@@@@@@@@@@', rmse)

        # print('""""""""""""""""""', np.unique(depth_d))
        rmse_log = (np.log(depth_g) - np.log(depth_d))**2
        rmse_log = np.sqrt(rmse_log.mean())




        abs_rel = np.mean(np.abs(depth_d - depth_g)/depth_g)

        # abs_rel_ = np.abs(depth_d - depth_g)/depth_g
        # abs_rel_ = abs_rel_[np.logical_not(np.isinf(abs_rel_))]
        # abs_rel = np.mean(abs_rel_)
        sq_rel = np.mean(((depth_g-depth_d)**2)/depth_g)
        # sq_rel_ = (depth_g-depth_d)**2/depth_g
        # sq_rel_ = sq_rel_[np.logical_not(np.isinf(sq_rel_))]
        # sq_rel = np.mean(sq_rel_)

        log10_error = np.abs(np.log10(depth_g)-np.log10(depth_d))
        log10_mean = np.mean(log10_error)
        metrics = [abs_rel, sq_rel, rmse, rmse_log, log10_mean, a1, a2, a3]
        return metrics


    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        # print('3333333333333333', dt)

        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]


        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        # print(p.iouType)
        # print('222222222222222', len(g))  # 1 or 0
        # if p.iouType == 'segm':
        #     print('33333333333333333', d[0]) # len(d) 0-26?
        # d: list (8,1,28,28)
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        def _summarize_Depth():
            stats = np.zeros((8,))
            for i in range(8):
                stats[i] = self.mean_error[i]
            return stats

        if not self.eval and self.params.iouType != 'depth' and self.params.iouType != 'whole_depth':
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        elif iouType == 'depth' or iouType == 'whole_depth':
            summarize = _summarize_Depth
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:
    def __init__(self, iouType='segm'):
        if iouType == 'depth':
            self.setDepthParams()
        elif iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        elif iouType == 'whole_depth':
            self.set_whole_depth_params()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

    def set_whole_depth_params(self):
        self.imgIds = []
        self.catIds = []
        self.useCats = 1
        self.maxDets = [1, 10, 100]

    def setDepthParams(self):
        self.imgIds = []
        self.catIds = []
        self.maxDets = [1, 10, 100]
        self.useCats = 1
        # self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        # self.areaRngLbl = ['all', 'small', 'medium', 'large']


    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
