import torch
import EmbCache
import time
import EmbCacheLib

len = 100_000

src_data = torch.randn((4000_000, 128)).float()
dst_data = torch.randn((400_000, 128)).float()

src_index = torch.randperm(4000_000).long()[:len]
dst_index = torch.randperm(400_000).long()[:len]

for i in range(1):
    begin = time.time()
    dst_data[dst_index] = src_data[src_index]
    end = time.time()
    print((end - begin) * 1000)

a = dst_data.clone()

EmbCacheLib.cudaPin(src_data)
dst_data = dst_data.cuda()
src_index = src_index.cuda()
dst_index = dst_index.cuda()

print()

for i in range(1):
    begin = time.time()
    EmbCacheLib.cudaFetch(src_data, src_index, dst_data, dst_index)
    torch.cuda.synchronize()
    end = time.time()
    print((end - begin) * 1000)

b = dst_data.clone()

print(torch.equal(a.cpu(), b.cpu()))
