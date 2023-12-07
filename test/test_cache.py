import torch
import EmbCache
import sys
import torch.distributed as dist

dist.init_process_group(backend="gloo", init_method="env://")

rank = dist.get_rank()
world_size = dist.get_world_size()

assert world_size == 2

if rank == 0:
    cache = EmbCache.capi.IndexCache(30, "test_cache", True)
dist.barrier()
if rank != 0:
    cache = EmbCache.capi.IndexCache(30, "test_cache", False)

for i in range(4):
    dist.barrier()
    if rank == 0:
        print(cache.get_valid_entries_num())

    idx = torch.arange(i * 10, (i + 1) * 10)
    if rank == i % world_size:
        locs = cache.add(idx)
        print(locs)
        print(idx)
    dist.barrier()

    if rank == i % world_size:
        mask = torch.tensor([1, 0], dtype=torch.bool).repeat(5).reshape((10, ))
    else:
        mask = torch.tensor([0, 1], dtype=torch.bool).repeat(5).reshape((10, ))

    lookup_locs, lookup_ids = cache.lookup(idx[mask])
    for j in range(world_size):
        dist.barrier()
        if rank == j:
            print("Rank", j)
            print(lookup_locs)
            print(lookup_ids)

    if rank == (i + 1) % world_size:
        cache.reset(idx[mask])
    dist.barrier()

    lookup_locs, lookup_ids = cache.lookup(idx[mask])
    for j in range(world_size):
        dist.barrier()
        if rank == j:
            print("Rank", j)
            print(lookup_locs)
            print(lookup_ids)

dist.barrier()
if rank == 0:
    idx, locs = cache.getall(15)
    print("getall")
    print(locs)
    print(idx)
