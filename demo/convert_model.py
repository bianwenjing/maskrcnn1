import torch

dic = torch.load("/home/wenjing/storage/result/result_/no_last_layers_res_1024.pth", map_location=torch.device('cpu'))
model = {'model': dic['model']}
# torch.save(model,'/home/wenjing/storage/result/result_/model_result11_4.pth')
torch.save(model,'/home/wenjing/storage/result/result_/no_last_layers_res_1024.pth')
# d = torch.load("/Users/bianwenjing/Desktop/model111.pth", map_location=torch.device('cpu'))
# d.keys()