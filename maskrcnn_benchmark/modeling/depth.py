import numpy as np
import torch
import os
from torch.autograd import Variable
from .models.base_model import BaseModel
import sys
from .models import pytorch_DIW_scratch

class HGModel(BaseModel):
    def name(self):
        return 'HGModel'

    def __init__(self, opt):
        BaseModel.initialize(self, opt)

        #print("===========================================LOADING Hourglass NETWORK====================================================")
        model = pytorch_DIW_scratch.pytorch_DIW_scratch
        model= torch.nn.parallel.DataParallel(model, device_ids = [0])
        #model_parameters = self.load_network(model, 'G', 'best_vanila')
        model_parameters = self.load_network(model, 'G', 'best_generalization')
        model.load_state_dict(model_parameters)
        self.netG = model.cuda()
        #print(model)

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print(save_path)
        model = torch.load(save_path)
        return model
        # network.load_state_dict(torch.load(save_path))