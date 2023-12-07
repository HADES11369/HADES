import sys
import torch
import EmbCache
import time

assert len(sys.argv) == 3
client_name = sys.argv[1]
server_name = sys.argv[2]

num_nodes = 50
emb_dim = 10
cache_rate = 0.2
cache_num = int(num_nodes * cache_rate)
idx_dtype = torch.int64
emb_dtype = torch.float32
emb = torch.arange(0, 500, dtype=emb_dtype).reshape(50, 10)

# test init
client = EmbCache.EmbCacheClient(server_name, client_name)
cache_name = server_name + "_cache"
cache_shape = [cache_num, emb_dim]
client.register_cache(cache_name, cache_shape, idx_dtype, emb_dtype)

for i in range(10):
    idx = torch.randint(0, num_nodes, (10, )).unique()
    cached_mask, idx_in_cache = client.split_idx(idx)
    uncached_idx = idx[~cached_mask]
    cached_idx = idx[cached_mask]
    print(idx)
    print(uncached_idx)
    print(cached_idx)
    print(cached_idx.numel())
    print(idx_in_cache)

    if uncached_idx.numel() > 0:
        uncached_emb = emb[uncached_idx]
    if cached_idx.numel() > 0:
        cached_emb = client.fetch_cached_embedding(idx_in_cache)

    result = torch.zeros((idx.numel(), emb_dim), dtype=emb_dtype)
    if uncached_idx.numel() > 0:
        print(cached_mask)
        result[~cached_mask] = uncached_emb
    if cached_idx.numel() > 0:
        result[cached_mask] = cached_emb
    print(result)

    if uncached_idx.numel() > 0:
        client.send_uncache_embedding(uncached_idx, uncached_emb)

    time.sleep(1)
