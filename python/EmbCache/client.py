from .common import *
from .cache import LocalCache
from .base import EmbCacheBase
import time
from multiprocessing.connection import Client


class EmbCacheClient(EmbCacheBase):

    def __init__(self, addr, port, local_rank, local_world_size) -> None:
        super().__init__(pin_memory=True)
        self.addr = addr
        self.port = port

        self._idx_cache = None
        self._idx_dtype = None
        self._emb_dtype = None
        self._emb_dim = None

        self._iter_cnt = 0

        self.local_world_size = local_world_size
        self.local_rank = local_rank
        self._is_chief = self.local_rank == 0

        self._transfer_stream = torch.cuda.Stream(priority=-1)
        self._transfer_remote_event = torch.cuda.Event()
        self._transfer_local_event = torch.cuda.Event()

        self._cpu2gpu_transfer_event1 = torch.cuda.Event()
        self._cpu2gpu_transfer_event2 = torch.cuda.Event()

        self._transfer_remote_status = True
        self._transfer_local_status = True

        self._transfer_remote_name = None
        self._transfer_local_name = None

        # socket connection
        self._socket_connection = None
        self._build_connection()

    def __del__(self):
        super().__del__()
        self._socket_connection.send(STOP_MSG)
        self._socket_connection.close()

    def _build_connection(self):
        print("Client {} build_connection start.".format(self.local_rank))

        self._socket_connection = Client((self.addr, self.port))
        self._socket_connection.send(self.local_rank)

        print("Client {} build_connection succeed.".format(self.local_rank))

    def _send_shm_tensor(self, tensor_name: str):
        if tensor_name != EMPTY_TENSOR_NAME:
            meta = self._shm_tensor_meta[tensor_name]
            msg = (tensor_name, meta.shm_name, meta.shm_size, meta.dtype,
                   meta.shape)
        else:
            msg = EMPTY_TENSOR_NAME
        self._socket_connection.send(msg)

    def _recv_shm_tensor(self):
        msg = self._socket_connection.recv()
        if msg == EMPTY_TENSOR_NAME:
            return None
        tensor_name, shm_name, shm_size, dtype, shape = msg
        return self._open_shm_tensor(tensor_name, shm_name, shm_size, dtype,
                                     shape)

    def register_cache(self, cache_name, cache_shape, idx_dtype, emb_dtype,
                       part_nnodes, part_start, hotness, cnt_threshold):
        # chief client send cache meta to server
        if self._is_chief:
            self._tmp_hotness = self._create_shm_tensor(
                'node_hotness', hotness.dtype, hotness.shape)
            self._tmp_hotness.copy_(hotness)
            cache_msg = (cache_name, cache_shape, idx_dtype, emb_dtype,
                         cnt_threshold)
            self._socket_connection.send(cache_msg)

            meta = self._shm_tensor_meta['node_hotness']
            msg = ('node_hotness', meta.shm_name, meta.shm_size, meta.dtype,
                   meta.shape)
            self._socket_connection.send(msg)

        time.sleep(1)

        self._idx_dtype = idx_dtype
        self._emb_dtype = emb_dtype
        self._emb_dim = cache_shape[1]

        # client wait to get shared cache buff tensor
        self._cache_buff = self._recv_shm_tensor()

        # open idx cache (only maintain node ids)
        self._idx_cache = LocalCache(cache_name,
                                     cache_shape[0],
                                     create=False,
                                     pin_memory=self._pin_memory)

        if self._is_chief:
            self._socket_connection.send(part_nnodes)
            self._socket_connection.send(part_start)

        # checkout register_cache finished
        status = self._socket_connection.recv()
        assert status

        print("Client register_cache succeed.".format())

    def split_idx(self, ids):
        # split ids into cached & uncached
        idx_in_cache, _ = self._idx_cache.lookup(ids)
        cached_mask = idx_in_cache != self._idx_cache.get_num_entries()
        return cached_mask, idx_in_cache[cached_mask]

    def fetch_cached_embedding(self, ids_in_cache):
        # fetch embedding from cache according to ids
        return self._cache_buff[ids_in_cache]

    def send_uncache_embedding(self, ids_name, ids, embs_name, embs):
        if ids is not None:
            self._send_shm_tensor(ids_name)
        else:
            self._send_shm_tensor(EMPTY_TENSOR_NAME)

        if embs is not None:
            self._send_shm_tensor(embs_name)

        else:
            self._send_shm_tensor(EMPTY_TENSOR_NAME)

    def create_send_cold_gradients_buffer(self, cold_remote_idx,
                                          cold_remote_grad, cold_local_idx,
                                          cold_local_grad):
        cold_idx_send_buff_name = EMPTY_TENSOR_NAME
        cold_grad_send_buff_name = EMPTY_TENSOR_NAME
        if cold_remote_idx.numel() > 0:
            with torch.cuda.stream(self._transfer_stream):
                cold_idx_send_buff_name = ("_cold_remote_grad_id_client_" +
                                           str(self.local_rank))
                cold_idx_send_buff = self._create_shm_tensor(
                    cold_idx_send_buff_name,
                    dtype=cold_remote_idx.dtype,
                    shape=cold_remote_idx.shape)
                cold_idx_send_buff.copy_(cold_remote_idx, non_blocking=True)

                cold_grad_send_buff_name = ("_cold_remote_grad_data_client_" +
                                            str(self.local_rank))
                cold_grad_send_buff = self._create_shm_tensor(
                    cold_grad_send_buff_name,
                    dtype=cold_remote_grad.dtype,
                    shape=cold_remote_grad.shape)
                cold_grad_send_buff.copy_(cold_remote_grad, non_blocking=True)

            self._transfer_remote_event.record(self._transfer_stream)

        self._transfer_remote_status = False
        self._transfer_remote_name = (cold_idx_send_buff_name,
                                      cold_grad_send_buff_name)

        cold_local_idx_send_buff_name = EMPTY_TENSOR_NAME
        cold_local_grad_send_buff_name = EMPTY_TENSOR_NAME
        if cold_local_idx.numel() > 0:
            with torch.cuda.stream(self._transfer_stream):
                cold_local_idx_send_buff_name = (
                    "_cold_local_grad_id_client_" + str(self.local_rank))
                cold_local_idx_send_buff = self._create_shm_tensor(
                    cold_local_idx_send_buff_name,
                    dtype=cold_local_idx.dtype,
                    shape=cold_local_idx.shape)
                cold_local_idx_send_buff.copy_(cold_local_idx,
                                               non_blocking=True)

                cold_local_grad_send_buff_name = (
                    "_cold_local_grad_data_client_" + str(self.local_rank))
                cold_local_grad_send_buff = self._create_shm_tensor(
                    cold_local_grad_send_buff_name,
                    dtype=cold_local_grad.dtype,
                    shape=cold_local_grad.shape)
                cold_local_grad_send_buff.copy_(cold_local_grad,
                                                non_blocking=True)

            self._transfer_local_event.record(self._transfer_stream)

        self._transfer_local_status = False
        self._transfer_local_name = (
            cold_local_idx_send_buff_name,
            cold_local_grad_send_buff_name,
        )

        self.try_send_cold_grads()

    def try_send_cold_grads(self):
        # remote first then local
        if self._transfer_remote_status == False:
            if self._transfer_remote_event.query():
                self._send_shm_tensor(self._transfer_remote_name[0])
                self._send_shm_tensor(self._transfer_remote_name[1])
                self._transfer_remote_name = None
                self._transfer_remote_status = True

        if self._transfer_local_status == False:
            if self._transfer_local_event.query():
                self._send_shm_tensor(self._transfer_local_name[0])
                self._send_shm_tensor(self._transfer_local_name[1])
                self._transfer_local_name = None
                self._transfer_local_status = True

    def force_send_cold_grads(self):
        # remote first then local
        if self._transfer_remote_status == False:
            self._transfer_remote_event.synchronize()
            self._send_shm_tensor(self._transfer_remote_name[0])
            self._send_shm_tensor(self._transfer_remote_name[1])
            self._transfer_remote_name = None
            self._transfer_remote_status = True

        if self._transfer_local_status == False:
            self._transfer_local_event.synchronize()
            self._send_shm_tensor(self._transfer_local_name[0])
            self._send_shm_tensor(self._transfer_local_name[1])
            self._transfer_local_name = None
            self._transfer_local_status = True

        self._iter_cnt += 1

    def get_staled_gradients(self, device):
        remote_idx = self._recv_shm_tensor()
        remote_grad = self._recv_shm_tensor()
        remote_cnt = self._recv_shm_tensor()
        if remote_idx is None or remote_grad is None or remote_cnt is None:
            part_remote_idx = torch.zeros((0, ),
                                          dtype=self._idx_dtype,
                                          device=device)
            part_remote_grad = torch.zeros((0, self._emb_dim),
                                           dtype=self._emb_dtype,
                                           device=device)
            part_remote_cnt = torch.zeros((0, ),
                                          dtype=torch.int64,
                                          device=device)
        else:
            with torch.cuda.stream(self._transfer_stream):
                part_remote_idx = remote_idx.to(device, non_blocking=True)
                part_remote_grad = remote_grad.to(device, non_blocking=True)
                part_remote_cnt = remote_cnt.to(device, non_blocking=True)
            self._cpu2gpu_transfer_event1.record(self._transfer_stream)

        local_idx = self._recv_shm_tensor()
        local_grad = self._recv_shm_tensor()
        local_cnt = self._recv_shm_tensor()
        if local_idx is None or local_grad is None or local_cnt is None:
            part_local_idx = torch.zeros((0, ),
                                         dtype=self._idx_dtype,
                                         device=device)
            part_local_grad = torch.zeros((0, self._emb_dim),
                                          dtype=self._emb_dtype,
                                          device=device)
            part_local_cnt = torch.zeros((0, ),
                                         dtype=torch.int64,
                                         device=device)
        else:
            with torch.cuda.stream(self._transfer_stream):
                part_local_idx = local_idx.to(device, non_blocking=True)
                part_local_grad = local_grad.to(device, non_blocking=True)
                part_local_cnt = local_cnt.to(device, non_blocking=True)
            self._cpu2gpu_transfer_event2.record(self._transfer_stream)

        return (
            part_remote_idx,
            part_remote_grad,
            part_remote_cnt,
            self._cpu2gpu_transfer_event1,
        ), (
            part_local_idx,
            part_local_grad,
            part_local_cnt,
            self._cpu2gpu_transfer_event2,
        )
