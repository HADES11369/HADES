import torch
import EmbCache
import time

cache = EmbCache.capi.IndexCache(4000_000, "test_cache", True)

torch.manual_seed(42)

print(cache.get_valid_entries_num())

for i in range(10):
    index = torch.randint(0, 4000_000, (2000_000, ))
    locs = cache.add(index)

# (todo) bad performance for cuda lookup
for i in range(10):
    index = torch.randint(0, 4000_000, (400_000, ))
    begin = time.time()
    out1 = cache.lookup(index)
    end = time.time()
    print("cpu lookup", (end - begin) * 1000)

    index = index.cuda()
    torch.cuda.synchronize()

    begin = time.time()
    out2 = cache.cudalookup(index)
    torch.cuda.synchronize()
    end = time.time()
    print("gpu lookup", (end - begin) * 1000)

    print(torch.equal(out1.cuda(), out2))
    print()
