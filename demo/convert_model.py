import torch

dic = torch.load("/home/wenjing/storage/result/result_nyu12/model_0360000.pth", map_location=torch.device('cpu'))
model = {'model': dic['model']}
torch.save(model,'/home/wenjing/storage/result/result_/0360000_result12_convert.pth')
# torch.save(model,'/home/wenjing/storage/result/result_/no_last_layers_res_1024.pth')
# d = torch.load("/Users/bianwenjing/Desktop/model111.pth", map_location=torch.device('cpu'))
# d.keys()