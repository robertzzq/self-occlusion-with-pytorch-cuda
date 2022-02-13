import torch
import texture_mapping

v1 = torch.FloatTensor([[0,0,0], [0,1,0], [1,0,0]]).cuda()
t1 = torch.FloatTensor([[0,0], [0,1], [1,0]]).cuda()
sample = torch.FloatTensor([[0.8, 0.2999999]]).cuda()

hehe = texture_mapping.texture_mapping(v1, t1, sample)
print(hehe)
