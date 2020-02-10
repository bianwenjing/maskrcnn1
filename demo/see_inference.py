import maskrcnn_benchmark
import torch
import sys


category = str(sys.argv[1])
path = '/home/wenjing/storage/result/result11_3/inference/ScanNet_val/' + category +'.pth'
model = torch.load(path)
print(model)