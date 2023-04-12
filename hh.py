import numpy as np
import torch
import torch.nn.functional as F
#
a=torch.FloatTensor([
    [1,2,3,4,0,0],
    [1,2,3,0,0,0]
])
print(a)
b=F.softmax(a,-1)
print(b)
# c=torch.mean(a,1)
# print(c)
#
# pad_attn_mask = a.data.eq(0).unsqueeze(1)  # b_size x 1 x len_k
# pad_attn_mask=pad_attn_mask.expand(2, 6, 6)
# print(a.size())
# print(pad_attn_mask)
# print(pad_attn_mask.size())
#
# c=np.random.uniform(-1,1,100)
# print(c)
# c=torch.FloatTensor(c)
# c=torch.nn.Parameter(c)
# print(c)

# glove_weight=np.load('data/glove_300.npy')
#
# print('sss')
a=[[1],[1]]
b=[[1]]
a.extend(b)
print(a)

