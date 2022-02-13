import torch
import occlusion_detection


v1 = torch.FloatTensor([[0,-2,0], [0,2,0], [2,0,0]]).cuda()
start = torch.FloatTensor([0,0,1]).cuda()
sample = torch.FloatTensor([[1,0,-1], [0.5, 0, 0.5]]).cuda()
ind = torch.IntTensor([10]).cuda()
final_res = occlusion_detection.occlusion_detection(v1, start, sample, ind)

print(final_res)