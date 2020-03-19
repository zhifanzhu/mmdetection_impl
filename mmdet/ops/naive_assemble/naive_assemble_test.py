import torch
from naive_assemble import NaiveAssemble

aff = torch.ones([2, 3*3, 4, 5], dtype=torch.float32).cuda()
feat = torch.ones([2, 3, 4, 5], dtype=torch.float32).cuda()

assemble = NaiveAssemble(k=1)
feat2 = assemble(aff, feat)
