import torch

dic = torch.load("/home/wenjing/storage/result/result11_3/model_0720000.pth", map_location=torch.device('cpu'))
model = {'model': dic['model']}
torch.save(model,'/home/wenjing/storage/result/result_/model_result11_3.pth')

# d = torch.load("/Users/bianwenjing/Desktop/model111.pth", map_location=torch.device('cpu'))
# d.keys()