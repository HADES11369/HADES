import torch
import EmbCache
import time
import EmbCacheLib

len = 100_000

src_data = torch.randn((4000_000, 128)).float()
dst_data = torch.randn((len, 128)).float().cuda()

cached_idx = torch.randperm(2000_000).long()
idx_map = torch.full((4000_000, ), -1)
idx_map[cached_idx] = torch.arange(0, 2000_000)
idx_map = idx_map.cuda()
gpu_src_data = src_data[cached_idx].cuda()

src_index = torch.randperm(4000_000).long()[:len].cuda()

# orig
for i in range(5):
    begin = time.time()
    dst_data[:] = src_data[src_index.cpu()].cuda()
    end = time.time()
    print((end - begin) * 1000)

a = dst_data.clone()

print()

# pytorch fetch for cached
for i in range(5):
    begin = time.time()
    lookup_idx = idx_map[src_index]
    cached_mask = lookup_idx != -1
    dst_data[cached_mask] = gpu_src_data[lookup_idx[cached_mask]]
    dst_data[~cached_mask] = src_data[src_index[~cached_mask].cpu()].cuda()
    torch.cuda.synchronize()
    end = time.time()
    print((end - begin) * 1000)

d = dst_data.clone()

print(torch.equal(a, d))

print()

EmbCacheLib.cudaPin(src_data)

# uva for cached
for i in range(5):
    begin = time.time()
    EmbCacheLib.cudaFetchWithCache2(src_data, gpu_src_data, src_index, idx_map,
                                    dst_data)
    torch.cuda.synchronize()
    end = time.time()
    print((end - begin) * 1000)

b = dst_data.clone()

print(torch.equal(a, b))

print()

# uva for no cache
for i in range(5):
    begin = time.time()
    EmbCacheLib.cudaFetch2(src_data, src_index, dst_data)
    torch.cuda.synchronize()
    end = time.time()
    print((end - begin) * 1000)

c = dst_data.clone()

print(torch.equal(a, c))

print()
