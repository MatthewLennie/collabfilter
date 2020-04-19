import torch

tensor_a = torch.tensor([[[1,2,3,4,5]],[[1,2,3,4,5]]])
tensor_b =  torch.tensor([[[1],[2],[3],[4],[5]],[[1],[2],[3],[4],[5]]])

print(tensor_a.matmul(tensor_b))

tensor_c = torch.tensor([1,1])