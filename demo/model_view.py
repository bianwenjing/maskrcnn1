import torch

dictionary = torch.load("/home/wenjing/storage/result/result26/model_0030000.pth")

for key in dictionary["model"].keys():
    print(key)

print(dictionary["model"]["whole_depth.feature_extractor.whole_depth_fcn1.0.weight"])