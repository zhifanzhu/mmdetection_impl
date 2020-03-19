from time import time
import torch
from naive_assemble import NaiveAssemble

def run():
    aff = torch.ones([2, 3*3, 4, 5], dtype=torch.float32).cuda()
    feat = torch.ones([2, 3, 4, 5], dtype=torch.float32).cuda()

    assemble = NaiveAssemble(k=1)
    a = time()
    print("Start assemble...")
    feat2 = assemble(aff, feat)
    print("Finished takes(ms): ", (time() - a) * 1000)

    print(feat2) # Should be all ones, since we have normalization

if __name__ == '__main__':
    run()
