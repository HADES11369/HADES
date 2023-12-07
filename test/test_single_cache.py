import torch
import EmbCache
import sys
import torch.distributed as dist
import time

cache = EmbCache.capi.IndexCache(4000_000, "test_cache", True)

torch.manual_seed(42)

print(cache.get_valid_entries_num())

for i in range(10):
    index = torch.randint(0, 4000_000, (200_000, ))
    # index = torch.tensor([1,1,1,1,1]).long()

    begin = time.time()
    locs = cache.add(index)
    end = time.time()
    print("insert", (end - begin) * 1000)
    # print(locs)

    begin = time.time()
    index = torch.randint(0, 4000_000, (200_000, ))
    out = cache.lookup(index)
    end = time.time()
    print("lookup", (end - begin) * 1000)

    begin = time.time()
    index = torch.randint(0, 4000_000, (20_000, ))
    out = cache.reset(index)
    end = time.time()
    print("reset", (end - begin) * 1000)

print(cache.get_valid_entries_num())
a, b = cache.getall(4000_000)
print(a, a.sum(), a.numel())
print(b, b.sum(), b.numel())
