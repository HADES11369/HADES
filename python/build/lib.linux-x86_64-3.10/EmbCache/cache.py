import torch
import EmbCacheLib
from .common import *


class LocalCache(object):

    def __init__(self,
                 name: str,
                 cache_size: int,
                 create: bool,
                 pin_memory: bool = False):
        super(LocalCache, self).__init__()
        self._cache_size = cache_size
        self._idx_cache = EmbCacheLib.IndexCache(cache_size, name, create,
                                                 pin_memory)

    def lookup(self, ids):
        return self._idx_cache.lookup(ids)

    def cudalookup(self, ids):
        return self._idx_cache.cudalookup(ids)

    def add(self, ids):
        return self._idx_cache.add(ids)

    def get_num_entries(self):
        return self._idx_cache.get_size()

    def reset(self, ids):
        self._idx_cache.reset(ids)


class LocalSparseAdamStateCache(object):

    def __init__(self,
                 emb,
                 state_step,
                 state_mem,
                 state_power,
                 count_hit=False):
        self.local_state = (state_step.local_partition,
                            state_mem.local_partition,
                            state_power.local_partition)
        for tensor in self.local_state:
            EmbCacheLib.cudaPin(tensor)

        self._partition_start = emb._partition_start
        self._hotness = emb._hotness
        self._local_num = self.local_state[0].numel()
        self._emb = emb

        # cache
        self.cached_state = None
        self.idx_map = None
        self.has_cached = False
        self._hit_num = 0
        self._lookup_num = 0

        self._count_hit = count_hit

    def __del__(self):
        for tensor in self.local_state:
            EmbCacheLib.cudaUnpin(tensor)

    def __getitem__(self, index):
        '''
        index is a gpu tensor
        and return gpu tensors (step, mem, power)
        '''
        step = torch.empty((index.numel(), ),
                           device='cuda',
                           dtype=torch.float32)
        mem = torch.empty((index.numel(), self.local_state[1].shape[1]),
                          device='cuda',
                          dtype=torch.float32)
        power = torch.empty((index.numel(), self.local_state[2].shape[1]),
                            device='cuda',
                            dtype=torch.float32)

        local_index = index - self._partition_start
        if self.has_cached:
            EmbCacheLib.cudaFetchWithCache2(self.local_state[0],
                                            self.cached_state[0], local_index,
                                            self.idx_map, step)
            EmbCacheLib.cudaFetchWithCache2(self.local_state[1],
                                            self.cached_state[1], local_index,
                                            self.idx_map, mem)
            EmbCacheLib.cudaFetchWithCache2(self.local_state[2],
                                            self.cached_state[2], local_index,
                                            self.idx_map, power)

            if self._count_hit:
                self._hit_num += (self.idx_map[local_index]
                                  != -1).nonzero().numel()
                self._lookup_num += local_index.numel()

            return (step, mem, power)

        else:
            EmbCacheLib.cudaFetch2(self.local_state[0], local_index, step)
            EmbCacheLib.cudaFetch2(self.local_state[1], local_index, mem)
            EmbCacheLib.cudaFetch2(self.local_state[2], local_index, power)
            return (step, mem, power)

    def __setitem__(self, index, value):
        '''
        index is a gpu tensor
        value are gpu tensors (step, mem, power)
        '''
        local_index = index - self._partition_start
        if self.has_cached:
            EmbCacheLib.cudaSetWithCache(value[0], self.local_state[0],
                                         self.cached_state[0], local_index,
                                         self.idx_map)
            EmbCacheLib.cudaSetWithCache(value[1], self.local_state[1],
                                         self.cached_state[1], local_index,
                                         self.idx_map)
            EmbCacheLib.cudaSetWithCache(value[2], self.local_state[2],
                                         self.cached_state[2], local_index,
                                         self.idx_map)

            if self._count_hit:
                self._hit_num += (self.idx_map[local_index]
                                  != -1).nonzero().numel()
                self._lookup_num += local_index.numel()

        else:
            EmbCacheLib.cudaSet(value[0], self.local_state[0], local_index)
            EmbCacheLib.cudaSet(value[1], self.local_state[1], local_index)
            EmbCacheLib.cudaSet(value[2], self.local_state[2], local_index)

    def create_cache(self, alloca_mem_size):

        mem_used = 0

        info = "========================================\n"
        info += "Rank {} builds state cache in GPU\n".format(self._emb._rank)

        idx_map_size = dtype_sizeof(self._emb._idx_dtype) * self._local_num
        alloca_mem_size -= idx_map_size

        info += "GPU alloca_mem_size {:.3f} GB\n".format(alloca_mem_size /
                                                         1024 / 1024 / 1024)

        local_ids = torch.arange(
            self._local_num,
            dtype=self._emb._idx_dtype) + self._partition_start
        part_local_ids = local_ids[local_ids %
                                   self._emb._world_size == self._emb._rank]

        item_size = self.local_state[0].element_size(
        ) + self.local_state[1].element_size(
        ) * self.local_state[1].shape[1] + self.local_state[2].element_size(
        ) * self.local_state[2].shape[1]
        cached_num = min(alloca_mem_size // item_size, part_local_ids.numel())

        info += "Local node num {}\n".format(part_local_ids.numel())
        info += "GPU cached node num {}\n".format(cached_num)

        # begin cache
        if cached_num > 0:
            idx_map = torch.full((self._local_num, ), -1)

            part_local_hotness = self._hotness[part_local_ids].clone().detach()
            cached_order = torch.argsort(part_local_hotness, descending=True)
            cached_index = part_local_ids[
                cached_order][:cached_num] - self._partition_start

            idx_map[cached_index] = torch.arange(0, cached_num)
            self.idx_map = idx_map.cuda()

            cached_state = []
            for i, v in enumerate(self.local_state):
                cached_v = v[cached_index].cuda()
                cached_state.append(cached_v)

            self.cached_state = tuple(cached_state)

            self.has_cached = True
            mem_used = cached_num * item_size + idx_map_size

        else:
            self.has_cached = False
            mem_used = 0

        self._hit_num = 0
        self._lookup_num = 0
        info += "GPU cache size {:.3f} GB\n".format(mem_used / 1024 / 1024 /
                                                    1024)
        info += "========================================\n"
        print(info)

        return mem_used

    def get_hit_rate(self):
        if self._lookup_num == 0:
            return 0
        return self._hit_num / self._lookup_num
