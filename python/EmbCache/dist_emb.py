import torch
import torch.nn as nn
from dgl.distributed.dist_tensor import DistTensor
from .common import tensor_initializer, EMPTY_TENSOR_NAME
from .client import EmbCacheClient
import EmbCacheLib
import time
from dgl import backend as F


class DistEmb(nn.Module):

    def __init__(self,
                 num_nodes,
                 emb_dim,
                 emb_dtype,
                 name,
                 cache_rate=0.3,
                 idx_dtype=None,
                 local_rank=None,
                 local_trainer_num=None,
                 hotness_list=None,
                 hot_threshold=150,
                 part_book=None,
                 cache_server_addr=None,
                 cache_server_port=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.sparse_emb = DistEmbeddingTensor(
            num_nodes,
            emb_dim,
            emb_dtype,
            name,
            cache_rate=cache_rate,
            idx_dtype=idx_dtype,
            local_rank=local_rank,
            local_trainer_num=local_trainer_num,
            hotness_list=hotness_list,
            hot_threshold=hot_threshold,
            part_book=part_book,
            cache_server_addr=cache_server_addr,
            cache_server_port=cache_server_port)

        self._device = torch.device("cuda")

    def forward(self, idx):
        return self.sparse_emb(idx, self._device, self.training)

    def timecheck(self):
        # self.sparse_emb.timecheck()
        pass


class DistEmbeddingTensor:

    def __init__(self,
                 num_nodes,
                 emb_dim,
                 emb_dtype,
                 name,
                 cache_rate=0.3,
                 idx_dtype=None,
                 local_rank=None,
                 local_trainer_num=None,
                 hotness_list=None,
                 hot_threshold=150,
                 part_book=None,
                 cache_server_addr=None,
                 cache_server_port=None):

        assert cache_server_addr is not None
        assert cache_server_port is not None

        self.tensor = DistTensor((num_nodes, emb_dim),
                                 emb_dtype,
                                 name=name,
                                 init_func=tensor_initializer)

        self._local_embedding_partition = self.tensor.local_partition
        # cuda pin
        EmbCacheLib.cudaPin(self._local_embedding_partition)
        self._partid = part_book.partid
        self._partition_start = (part_book._max_node_ids[self._partid - 1]
                                 if self._partid > 0 else 0)
        self._hotness = hotness_list
        self._rank = local_rank
        self._world_size = local_trainer_num
        self._idx_dtype = idx_dtype

        self.part_book = part_book
        self._partition_range = [0]
        for i in range(part_book.num_partitions()):
            self._partition_range.append(part_book._max_node_ids[i])
        self._partition_range = torch.tensor(self._partition_range).cuda()

        self.name = name
        self.num_embeddings = num_nodes
        self.embedding_dim = emb_dim
        self._emb_dtype = emb_dtype

        self.enable_cache = cache_rate > 0
        if self.enable_cache:
            assert hotness_list is not None
            assert part_book is not None
            assert local_rank is not None
            assert local_trainer_num is not None

            self._cache_rate = cache_rate
            self._hot_mask = hotness_list >= hot_threshold

            self._local_mask = (part_book.nid2partid(
                torch.arange(num_nodes)) == self._partid)

            self._total_mask = torch.zeros(num_nodes).char()
            # local = 0
            self._total_mask[self._local_mask] = 0
            # remote hot = 1
            self._total_mask[self._hot_mask & (~self._local_mask)] = 1
            # remote cold = 2
            self._total_mask[(~self._hot_mask) & (~self._local_mask)] = 2

            self._total_mask = self._total_mask.cuda()

            self._server_name = name + "_server"
            self._client_name = name + "_client" + str(local_rank)
            self.client = EmbCacheClient(cache_server_addr, cache_server_port,
                                         local_rank, local_trainer_num)

            self._cache_name = name + "_cache"
            self._cache_shape = [int(num_nodes * cache_rate), emb_dim]
            self.client.register_cache(
                self._cache_name,
                self._cache_shape,
                idx_dtype,
                emb_dtype,
                part_book.metadata()[self._partid]["num_nodes"],
                self._partition_start,
                hotness_list,
                cnt_threshold=16,
            )

            self.shm_uncache_ids_name = "uncache_ids" + self._client_name
            self.shm_uncache_emb_name = "uncache_emb" + self._client_name

        self._reset_recorder()

        self.trace = []

    def _reset_recorder(self):
        self._time_split = 0.0
        self._fetch_local = 0.0
        self._fetch_remote_hot = 0.0
        self._fetch_cache_cold = 0.0
        self._fetch_uncache_cold = 0.0
        self._cudalookup = 0.0
        self._send_uncache = 0.0
        self._num_iters = 0

        self._local_size = 0.0
        self._remote_hot_size = 0.0
        self._remote_cold_uncache_size = 0.0
        self._remote_cold_cache_size = 0.0

    def __del__(self):
        EmbCacheLib.cudaUnpin(self._local_embedding_partition)

    def __call__(self, ids, device=torch.device("cuda"), is_training=True):
        if not is_training:
            return self.tensor[ids.cpu()].to(device)

        ids = ids.cuda()
        if self.enable_cache:
            tic = time.time()
            local_mask, remote_hot_mask, remote_cold_mask = EmbCacheLib.cudaSplit(
                ids, self._total_mask)

            # local_index = torch.nonzero(local_mask).squeeze()
            # remote_hot_index = torch.nonzero(remote_hot_mask).squeeze()
            dist_index = torch.nonzero(local_mask | remote_hot_mask).squeeze()
            remote_cold_index = torch.nonzero(remote_cold_mask).squeeze()

            tic = time.time()
            # todo (ping): accelerate lookup
            locs = self.client._idx_cache.lookup(
                ids[remote_cold_index].cpu()).cuda()
            in_cache_mask = locs >= 0
            # torch.cuda.synchronize()
            self._cudalookup += time.time() - tic

            in_cache_index = torch.nonzero(in_cache_mask).squeeze()
            not_in_cache_index = torch.nonzero(~in_cache_mask).squeeze()

            # self._local_size += local_index.numel()
            # self._remote_hot_size += remote_hot_index.numel()
            # self._remote_cold_cache_size += in_cache_index.numel()
            # self._remote_cold_uncache_size += not_in_cache_index.numel()

            # begin fetch
            emb = torch.empty((ids.shape[0], self.embedding_dim),
                              dtype=self._emb_dtype,
                              device=device)
            # torch.cuda.synchronize()
            self._time_split += time.time() - tic

            # fetch remote cold in cache
            tic = time.time()
            if in_cache_index.numel() > 0:
                # print("fetch cold cache")
                src_index = locs[in_cache_index]
                dst_index = remote_cold_index[in_cache_index]
                EmbCacheLib.cudaFetch(self.client._cache_buff, src_index, emb,
                                      dst_index)
            # torch.cuda.synchronize()
            self._fetch_cache_cold += time.time() - tic

            shm_uncache_ids = None
            shm_uncache_emb = None

            # fetch remote hot and uncache cold
            if dist_index.numel() > 0 or not_in_cache_index.numel() > 0:
                tic = time.time()
                ## create uncache_ids buffer
                uncache_cold_index = remote_cold_index[
                    not_in_cache_index].reshape(-1)
                uncache_cold_ids = ids[uncache_cold_index].reshape(-1)
                shm_uncache_ids = self.client._create_shm_tensor(
                    self.shm_uncache_ids_name, uncache_cold_ids.dtype,
                    uncache_cold_ids.shape)
                if shm_uncache_ids is not None:
                    shm_uncache_ids.copy_(uncache_cold_ids, non_blocking=False)

                ## create uncache_emb buffer
                shm_uncache_emb = self.client._create_shm_tensor(
                    self.shm_uncache_emb_name, self.tensor.dtype,
                    (uncache_cold_ids.shape[0], self.tensor.shape[1]))
                # torch.cuda.synchronize()
                self._fetch_uncache_cold += time.time() - tic

                tic = time.time()
                remote_ids = torch.cat([ids[dist_index], uncache_cold_ids])
                emb_dst_index = torch.cat([dist_index, uncache_cold_index])
                remote_emb = self.tensor[remote_ids.cpu()]
                emb[emb_dst_index] = remote_emb.to(device, non_blocking=True)
                # torch.cuda.synchronize()
                self._fetch_remote_hot += time.time() - tic

                ## copy uncache into shm_uncache_emb
                tic1 = time.time()
                if shm_uncache_emb is not None:
                    shm_uncache_emb[:] = remote_emb[dist_index.shape[0]:]
                self._fetch_uncache_cold += time.time() - tic1

            # send uncache embedding to server
            tic = time.time()
            self.client.send_uncache_embedding(
                self.shm_uncache_ids_name,
                shm_uncache_ids,
                self.shm_uncache_emb_name,
                shm_uncache_emb,
            )
            # torch.cuda.synchronize()
            self._send_uncache += time.time() - tic

        else:
            emb = self.tensor[ids].to(device)

        emb.requires_grad_(True)
        emb.retain_grad()

        self._num_iters += 1
        if self._num_iters % 10 == 0:
            pass
            # self.timecheck()
            # self._reset_recorder()

        if F.is_recording():
            emb = F.attach_grad(emb)
            if self.enable_cache:
                hot_mask = self._hot_mask[ids.cpu()].to(device,
                                                        non_blocking=True)
                self.trace.append((
                    ids.to(device, non_blocking=True),
                    emb,
                    hot_mask.to(device, non_blocking=True),
                    local_mask.to(device, non_blocking=True),
                ))
            else:
                self.trace.append((ids.to(device, non_blocking=True), emb))

        # self.client.get_staled_gradients()

        return emb

    def reset_trace(self):
        """Reset the traced data."""
        self.trace = []

    def timecheck(self):
        timetable = ("============DistEmb Time Log============\n"
                     "Iter {}\n"
                     "Split: {:.3f} ms\n"
                     "Fetch local: {:.3f} ms\n"
                     "Fetch remote hot: {:.3f} ms\n"
                     "Fetch cache cold: {:.3f} ms\n"
                     "Fetch uncache cold: {:.3f} ms\n"
                     "Cuda lookup: {:.3f} ms\n"
                     "Send uncache: {:.3f} ms\n"
                     "Local size: {:.3f} MB\n"
                     "Remote hot size: {:.3f} MB\n"
                     "Remote cold cache size: {:.3f} MB\n"
                     "Remote cold uncache size: {:.3f} MB\n"
                     "========================================".format(
                         self._num_iters,
                         self._time_split / self._num_iters * 1000,
                         self._fetch_local / self._num_iters * 1000,
                         self._fetch_remote_hot / self._num_iters * 1000,
                         self._fetch_cache_cold / self._num_iters * 1000,
                         self._fetch_uncache_cold / self._num_iters * 1000,
                         self._cudalookup / self._num_iters * 1000,
                         self._send_uncache / self._num_iters * 1000,
                         self._local_size * self.embedding_dim * 4 / 1024 /
                         1024,
                         self._remote_hot_size * self.embedding_dim * 4 /
                         1024 / 1024,
                         self._remote_cold_cache_size * self.embedding_dim *
                         4 / 1024 / 1024,
                         self._remote_cold_uncache_size * self.embedding_dim *
                         4 / 1024 / 1024,
                     ))
        print(timetable)
