from maskrcnn_benchmark.data.datasets.mydataset import ScanNetDataset
from predictor import COCODemo
from maskrcnn_benchmark.config import cfg

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False,
)

anno = '/home/wenjing/storage/anno/ground.txt'
root = '/home/wenjing/storage/ScanNetv2/test'
config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

a = ScanNetDataset(anno,root)
print(a)
ground_label = a(1)
print(ground_label)
# f = open('/home/wenjing/storage/ScanNetv2/test.txt', "r")
#
# result = coco_demo.run_on_ground_truth()