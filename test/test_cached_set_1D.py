import torch
import EmbCache
import time
import EmbCacheLib

len = 100_000

dst_data = torch.randn((4000_000, )).float()
src_data = torch.randn((len, )).float().cuda()

cached_idx = torch.randperm(2000_000).long()
idx_map = torch.full((4000_000, ), -1)
idx_map[cached_idx] = torch.arange(0, 2000_000)
idx_map = idx_map.cuda()
gpu_dst_data = dst_data[cached_idx].cuda()

dst_index = torch.randperm(4000_000).long()[:len].cuda()

# orig
for i in range(5):
    begin = time.time()
    dst_data[dst_index.cpu()] = src_data.cpu()
    end = time.time()
    print((end - begin) * 1000)

a = dst_data[dst_index.cpu()].clone()

print()

# pytorch set for cached
for i in range(5):
    begin = time.time()
    lookup_idx = idx_map[dst_index]
    cached_mask = lookup_idx != -1
    gpu_dst_data[lookup_idx[cached_mask]] = src_data[cached_mask]
    dst_data[dst_index[~cached_mask].cpu()] = src_data[~cached_mask].cpu()
    torch.cuda.synchronize()
    end = time.time()
    print((end - begin) * 1000)

d = dst_data[dst_index.cpu()].clone()

print(torch.equal(a, d))

EmbCacheLib.cudaPin(dst_data)

print()

# cached
for i in range(5):
    begin = time.time()
    EmbCacheLib.cudaSetWithCache(src_data, dst_data, gpu_dst_data, dst_index,
                                 idx_map)
    torch.cuda.synchronize()
    end = time.time()
    print((end - begin) * 1000)

b = dst_data[dst_index.cpu()].clone()

print(torch.equal(a, b))

print()

# uva
for i in range(5):
    begin = time.time()
    EmbCacheLib.cudaSet(src_data, dst_data, dst_index)
    torch.cuda.synchronize()
    end = time.time()
    print((end - begin) * 1000)

c = dst_data[dst_index.cpu()].clone()

print(torch.equal(a, c))
