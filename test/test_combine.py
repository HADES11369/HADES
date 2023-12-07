import torch
import EmbCache
import time
import EmbCacheLib

len = 100_000

dst_data = torch.zeros((4000_000, 128)).float()
src_data = torch.randn((400_000, 128)).float()

index = torch.randperm(4000_000).long()[:400_000].cuda()

for i in range(10):
    begin = time.time()
    #tmp = src_data.cuda()
    dst_data[index] = src_data
    end = time.time()
    print((end - begin) * 1000)

a = dst_data.clone()

dst_data[:] = 0

dst_data = dst_data.cuda()
# EmbCacheLib.cudaPin(src_data)
src_data = src_data.cuda()
index = index.cuda()

print()

for i in range(10):
    begin = time.time()
    EmbCacheLib.cudaCombine(src_data, dst_data, index)
    torch.cuda.synchronize()
    end = time.time()
    print((end - begin) * 1000)

b = dst_data.clone()

print(torch.equal(a.cuda(), b.cuda()))
