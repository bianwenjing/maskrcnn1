import torchvision
from .pycocotools2.coco2 import COCO2


class CocoDetection2(torchvision.datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoDetection2, self).__init__(root, annFile, transform, target_transform, transforms)
        self.coco = COCO2(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
