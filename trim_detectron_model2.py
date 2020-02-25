import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="~/.torch/models/e2e_mask_rcnn_R_50_C4_1x.pth",
    # default="~/.torch/models/_detectron_35858933_12_2017_baselines_e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC_output_train_coco_2014_train%3Acoco_2014_valminusminival_generalized_rcnn_model_final.pkl",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    # default="./pretrained_model/mask_rcnn_R-50-FPN_1x_detectron_no_last_layers.pth",
    # default="~/maskrcnn-anubis/no_last_layers.pth",
    default="/home/wenjing/storage/result/result_/no_last_layers_res_1024.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="configs/e2e_mask_rcnn_R_50_FPN_1x.yaml",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

# cfg.merge_from_file(args.cfg)

_d = torch.load(DETECTRON_PATH, map_location=torch.device('cpu'))

# _d = load_c2_format(cfg, DETECTRON_PATH)
newdict = _d

# print(newdict['model'].keys())

# newdict['model'] = removekey(_d['model'],
#                              ['cls_score.bias', 'cls_score.weight', 'bbox_pred.bias', 'bbox_pred.weight', 'mask_fcn_logits.bias', 'mask_fcn_logits.weight'])
newdict['model'] = removekey(_d['model'],
                              ['module.roi_heads.mask.predictor.mask_fcn_logits.weight',
                               'module.roi_heads.mask.predictor.mask_fcn_logits.bias',
                               'module.roi_heads.box.predictor.cls_score.weight',
                               'module.roi_heads.box.predictor.cls_score.bias',
                               'module.roi_heads.box.predictor.bbox_pred.weight',
                               'module.roi_heads.box.predictor.bbox_pred.bias',
                               'module.roi_heads.mask.predictor.conv5_mask.weight',
                               'module.roi_heads.mask.predictor.conv5_mask.bias'])
torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))