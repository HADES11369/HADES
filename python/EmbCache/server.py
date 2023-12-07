from .common import *
from .cache import LocalCache
import time
from .base import EmbCacheBase
from queue import Queue
from multiprocessing.connection import Listener


class EmbCacheServer(EmbCacheBase):

    def __init__(self, addr, port, num_clients, check_iters=50) -> None:
        super().__init__(pin_memory=False)

        self.addr = addr
        self.port = port
        self.num_clients = num_clients

        self._cache_name = None
        self._idx_dtype = None
        self._emb_dtype = None
        self._emb_dim = None

        self._idx_cache = None
        self._emb_cache_buff_name = None
        self._emb_cache_buff = None

        self._grad_cache_buff_name = None
        self._grad_cache_buff = None
        self._grad_cnt_cache_buff_name = None
        self._grad_cnt_cache_buff = None
        self._grad_idx_cache_buff_name = None
        self._grad_idx_cache_buff = None

        self._part_nnodes = None
        self._local_grad_buff = None
        self._local_grad_buff_name = None
        self._local_grad_cnt_buff = None
        self._local_grad_cnt_buff_name = None

        self._check_iters = check_iters

        self._reset_recorder()

        self._socket_listener = None
        self._socket_connections = []

        self._build_connection()

        self._create_cache()

        self._start()

    def _reset_recorder(self):
        self._iter_cnt = 0
        self._update_time_insert_cache = [0 for _ in range(self.num_clients)]
        self._update_time_insert_emb = [0 for _ in range(self.num_clients)]
        self._update_time_empty_grad = [0 for _ in range(self.num_clients)]
        self._num_ids_uncached = [0 for _ in range(self.num_clients)]
        self._num_ids_real_add = [0 for _ in range(self.num_clients)]

        self._time_recv_cold_grads = 0
        self._num_cold_remote_grads = [0 for _ in range(self.num_clients)]
        self._num_cold_local_grads = [0 for _ in range(self.num_clients)]

        self._accum_time_cold_remote = 0
        self._accum_time_cold_remote_update = 0
        self._accum_time_cold_local = 0
        self._accum_time_cold_local_update = 0

        self._time_send_remote_staled = 0
        self._time_send_local_staled = 0

    def __del__(self):
        pass

    def _create_cache(self):
        # revice infomation from chief client
        cache_msg = self._socket_connections[0].recv()
        cache_name, cache_shape, idx_dtype, emb_dtype, cnt_threshold = cache_msg
        hotness_meta = self._socket_connections[0].recv()

        # receive hot tensor
        tensor_name, shm_name, shm_size, dtype, shape = hotness_meta
        self.hotness = self._open_shm_tensor(tensor_name, shm_name, shm_size,
                                             dtype, shape)
        self.cnt_threshold = int(cnt_threshold)
        self.threshold = (1 /
                          (0.001 + self.hotness)).ceil().int().clamp(max=100)

        # create EmbeddingCache
        self._cache_name = cache_name
        self._idx_cache = LocalCache(cache_name,
                                     cache_shape[0],
                                     create=True,
                                     pin_memory=self._pin_memory)
        self._idx_dtype = idx_dtype
        self._emb_dtype = emb_dtype
        cache_shape[0] = self._idx_cache.get_num_entries()
        self._emb_dim = cache_shape[1]

        # create cache buffer for emb and send to client
        self._emb_cache_buff_name = cache_name + "_emb_buff"
        assert self._emb_cache_buff_name not in self._shm_tensor_meta
        self._emb_cache_buff = self._create_shm_tensor(
            self._emb_cache_buff_name, dtype=emb_dtype, shape=cache_shape)
        self._send_shm_tensor_to_all(self._emb_cache_buff_name)

        # create cache buffer for grads
        self._grad_cache_buff_name = cache_name + "_grad_buff"
        assert self._grad_cache_buff_name not in self._shm_tensor_meta
        self._grad_cache_buff = self._create_shm_tensor(
            self._grad_cache_buff_name, dtype=emb_dtype, shape=cache_shape)
        self._grad_cache_buff[:] = 0

        # create cache buffer for grads cnt
        self._grad_cnt_cache_buff_name = cache_name + "_grad_cnt_buff"
        assert self._grad_cnt_cache_buff_name not in self._shm_tensor_meta
        self._grad_cnt_cache_buff = self._create_shm_tensor(
            self._grad_cnt_cache_buff_name,
            dtype=torch.int32,
            shape=(cache_shape[0], ))
        self._grad_cnt_cache_buff[:] = 0

        # create cache buffer for grads iter count
        self._grad_iter_count = torch.zeros_like(
            self._grad_cnt_cache_buff).int()

        self._grad_iter_threshold = torch.empty_like(self._grad_iter_count)

        #################### for local grads ################################

        # create cache buffer for grads idx
        self._grad_idx_cache_buff_name = cache_name + "_grad_idx_buff"
        assert self._grad_idx_cache_buff_name not in self._shm_tensor_meta
        self._grad_idx_cache_buff = self._create_shm_tensor(
            self._grad_idx_cache_buff_name,
            dtype=self._idx_dtype,
            shape=(cache_shape[0], ))
        self._grad_idx_cache_buff[:] = -1

        # create buffers for local grads
        self._part_nnodes = int(self._socket_connections[0].recv())
        self._part_start = int(self._socket_connections[0].recv())

        self._local_grad_buff_name = cache_name + "_local_grad_buff"
        assert self._local_grad_buff_name not in self._shm_tensor_meta
        self._local_grad_buff = self._create_shm_tensor(
            self._local_grad_buff_name,
            dtype=self._emb_dtype,
            shape=(self._part_nnodes, cache_shape[1]))
        self._local_grad_buff[:] = 0

        self._local_grad_cnt_buff_name = cache_name + "_local_grad_cnt_buff"
        assert self._local_grad_cnt_buff_name not in self._shm_tensor_meta
        self._local_grad_cnt_buff = self._create_shm_tensor(
            self._local_grad_cnt_buff_name,
            dtype=torch.int32,
            shape=(self._part_nnodes, ))
        self._local_grad_cnt_buff[:] = 0

        # create cache buffer for local grads iter count
        self._local_grad_iter_count = torch.zeros_like(
            self._local_grad_cnt_buff).int()

        # create local grad iter threshold
        self._local_grad_iter_threshold = self.threshold[self._part_start:self.
                                                         _part_start +
                                                         self._part_nnodes]

        # send signal to all clients informing that the server is ready
        for conn in self._socket_connections:
            conn.send(True)

        print("Server create_cache succeed.")

    def _send_shm_tensor_to_all(self, tensor_name):
        msg = None
        if tensor_name != EMPTY_TENSOR_NAME:
            meta = self._shm_tensor_meta[tensor_name]
            msg = (tensor_name, meta.shm_name, meta.shm_size, meta.dtype,
                   meta.shape)
        else:
            msg = EMPTY_TENSOR_NAME

        for conn in self._socket_connections:
            conn.send(msg)

    def _send_shm_tensor_to_one(self, tensor_name, client_id):
        msg = None
        if tensor_name != EMPTY_TENSOR_NAME:
            meta = self._shm_tensor_meta[tensor_name]
            msg = (tensor_name, meta.shm_name, meta.shm_size, meta.dtype,
                   meta.shape)
        else:
            msg = EMPTY_TENSOR_NAME
        self._socket_connections[client_id].send(msg)

    def _recv_shm_tensor_from_all(self):
        received_tensors = []
        for conn in self._socket_connections:
            msg = conn.recv()
            if msg == STOP_MSG:
                return STOP_MSG

            if msg == EMPTY_TENSOR_NAME:
                received_tensors.append(None)
                continue

            tensor_name, shm_name, shm_size, dtype, shape = msg
            tensor = self._open_shm_tensor(tensor_name, shm_name, shm_size,
                                           dtype, shape)

            received_tensors.append(tensor)

        return received_tensors

    def _try_recv_shm_tensor_from_one(self, client_id):
        have = self._socket_connections[client_id].poll(0)
        if have:
            msg = self._socket_connections[client_id].recv()
            if msg == EMPTY_TENSOR_NAME:
                return None
            tensor_name, shm_name, shm_size, dtype, shape = msg
            return self._open_shm_tensor(tensor_name, shm_name, shm_size,
                                         dtype, shape)
        else:
            return ""

    def _recv_shm_tensor_from_one(self, client_id):
        msg = self._socket_connections[client_id].recv()
        if msg == EMPTY_TENSOR_NAME:
            return None
        tensor_name, shm_name, shm_size, dtype, shape = msg
        return self._open_shm_tensor(tensor_name, shm_name, shm_size, dtype,
                                     shape)

    def _build_connection(self):
        print("Server build_connection start.")

        self._socket_listener = Listener((self.addr, self.port))

        tmp_conns = []
        for _ in range(self.num_clients):
            conn = self._socket_listener.accept()
            tmp_conns.append(conn)
            print("Server accept client {}.".format(
                self._socket_listener.last_accepted))

        self._socket_connections = [None] * self.num_clients

        for conn in tmp_conns:
            client_id = conn.recv()
            self._socket_connections[client_id] = conn
        print("Server build_connection succeed.")

    def _start(self):
        self._loop()

    def _loop(self):
        """
        loop function to handle requests from clients
        """

        try:
            while True:
                # step 0:
                # print("begin send staled grads")
                self._send_staled_gradients()
                # print("end send staled grads")

                # step 1:
                uncache_ids, uncache_embs = self._wait_for_uncache_embedding()
                if uncache_ids == STOP_MSG:
                    break

                # step 2:
                # print("begin update data")
                self._update_cache(uncache_ids, uncache_embs)
                # print("end update data")

                # step 3:
                self._wait_for_cold_gradients()

                self._iter_cnt += 1

                if self._iter_cnt % self._check_iters == 0:
                    pass
                    # self._timecheck()
                    # self._reset_recorder()

                # if self._check_stop():
                #     break
        except EOFError:
            print("Server Exits")
            super().__del__()
        except ConnectionResetError:
            print("Server Exits")
            super().__del__()

    def _wait_for_uncache_embedding(self):
        recv_ids = self._recv_shm_tensor_from_all()
        recv_embs = self._recv_shm_tensor_from_all()
        if recv_ids == STOP_MSG:
            return STOP_MSG, STOP_MSG

        for i in range(self.num_clients):
            if recv_ids[i] is None or recv_embs[i] is None:
                recv_ids[i] = torch.zeros((0, ), dtype=self._idx_dtype)
                recv_embs[i] = torch.zeros((0, self._emb_dim),
                                           dtype=self._emb_dtype)
        return recv_ids, recv_embs

    def _wait_for_cold_gradients(self):
        polling_queue = Queue(self.num_clients)
        for i in range(self.num_clients):
            polling_queue.put(i)
        recv_cnt = [0 for _ in range(self.num_clients)]

        tic = time.time()
        while not polling_queue.empty():
            client_id = polling_queue.get()
            id = self._try_recv_shm_tensor_from_one(client_id)
            if id == "":
                polling_queue.put(client_id)
            else:
                grad = self._recv_shm_tensor_from_one(client_id)
                if id is None or grad is None:
                    id = torch.zeros((0, ), dtype=self._idx_dtype)
                    grad = torch.zeros((0, self._emb_dim),
                                       dtype=self._emb_dtype)

                toc = time.time()
                self._time_recv_cold_grads += toc - tic

                if recv_cnt[client_id] == 0:  # remote cold
                    self._num_cold_remote_grads[client_id] += id.shape[0]
                    self._accumulate_gradients_cold_remote(id, grad)
                    polling_queue.put(client_id)
                    recv_cnt[client_id] += 1
                else:  # local cold
                    self._num_cold_local_grads[client_id] += id.shape[0]
                    self._accumulate_gradients_cold_local(id, grad)

                tic = time.time()

    def _send_staled_gradients(self):
        # cold - remote - cached
        tic = time.time()

        remote_cnt_mask = self._grad_cnt_cache_buff > 0
        remote_cnt_index = torch.nonzero(remote_cnt_mask).flatten()
        remote_iter_mask = (self._grad_iter_count[remote_cnt_index]
                            >= self._grad_iter_threshold[remote_cnt_index]) | (
                                self._grad_cnt_cache_buff[remote_cnt_index]
                                >= self.cnt_threshold)

        remote_staled_index = remote_cnt_index[remote_iter_mask]
        remote_staled_num = remote_staled_index.numel()

        if remote_staled_num > 0:
            remote_staled_idx = self._grad_idx_cache_buff[remote_staled_index]
            remote_staled_grads = self._grad_cache_buff[remote_staled_index]
            remote_staled_cnt = self._grad_cnt_cache_buff[remote_staled_index]

            split = remote_staled_idx % self.num_clients

            for rank in range(self.num_clients):
                part_mask = split == rank
                part_index = part_mask.nonzero().flatten()
                part_size = part_index.shape[0]

                if part_size > 0:
                    staled_idx_send_buff_name = "_remote_staled_id_abc_" + str(
                        self._iter_cnt % 2) + "_" + str(rank)
                    staled_grad_send_buff_name = "_remote_staled_grad_abc_" + str(
                        self._iter_cnt % 2) + "_" + str(rank)
                    staled_grad_cnt_send_buff_name = "_remote_staled_grad_cnt_abc_" + str(
                        self._iter_cnt % 2) + "_" + str(rank)

                    staled_idx_send_buff = self._create_shm_tensor(
                        staled_idx_send_buff_name, self._idx_dtype,
                        (part_size, ))
                    staled_grad_send_buff = self._create_shm_tensor(
                        staled_grad_send_buff_name, self._emb_dtype,
                        (part_size, self._emb_dim))
                    staled_grad_cnt_send_buff = self._create_shm_tensor(
                        staled_grad_cnt_send_buff_name, torch.int64,
                        (part_size, ))

                    staled_idx_send_buff.copy_(remote_staled_idx[part_index])
                    staled_grad_send_buff.copy_(
                        remote_staled_grads[part_index])
                    staled_grad_cnt_send_buff.copy_(
                        remote_staled_cnt[part_index])

                    self._send_shm_tensor_to_one(staled_idx_send_buff_name,
                                                 rank)
                    self._send_shm_tensor_to_one(staled_grad_send_buff_name,
                                                 rank)
                    self._send_shm_tensor_to_one(
                        staled_grad_cnt_send_buff_name, rank)
                else:
                    self._send_shm_tensor_to_one(EMPTY_TENSOR_NAME, rank)
                    self._send_shm_tensor_to_one(EMPTY_TENSOR_NAME, rank)
                    self._send_shm_tensor_to_one(EMPTY_TENSOR_NAME, rank)

        else:
            self._send_shm_tensor_to_all(EMPTY_TENSOR_NAME)
            self._send_shm_tensor_to_all(EMPTY_TENSOR_NAME)
            self._send_shm_tensor_to_all(EMPTY_TENSOR_NAME)

        # reset cache and buffer
        self._idx_cache.reset(self._grad_idx_cache_buff[remote_staled_index])
        self._grad_cache_buff[remote_staled_index][:] = 0
        self._grad_cnt_cache_buff[remote_staled_index] = 0

        # update iter
        self._grad_iter_count[remote_cnt_index] += 1
        self._grad_iter_count[remote_staled_index] = 0

        toc = time.time()
        self._time_send_remote_staled += toc - tic

        # cold - local
        tic = time.time()

        local_cnt_mask = self._local_grad_cnt_buff > 0
        local_cnt_index = torch.nonzero(local_cnt_mask).flatten()
        local_iter_mask = (self._local_grad_iter_count[local_cnt_index]
                           >= self._local_grad_iter_threshold[local_cnt_index]
                           ) | (self._local_grad_cnt_buff[local_cnt_index]
                                >= self.cnt_threshold)

        local_staled_index = local_cnt_index[local_iter_mask]
        local_staled_num = local_staled_index.numel()

        if local_staled_num > 0:
            local_staled_idx = local_staled_index + self._part_start
            local_staled_grads = self._local_grad_buff[local_staled_index]
            local_staled_cnt = self._local_grad_cnt_buff[local_staled_index]

            split = local_staled_idx % self.num_clients

            for rank in range(self.num_clients):
                part_mask = split == rank
                part_index = part_mask.nonzero().flatten()
                part_size = part_index.shape[0]

                if part_size > 0:
                    staled_idx_send_buff_name = "_local_staled_id_abc_" + str(
                        self._iter_cnt % 2) + "_" + str(rank)
                    staled_grad_send_buff_name = "_local_staled_grad_abc_" + str(
                        self._iter_cnt % 2) + "_" + str(rank)
                    staled_grad_cnt_send_buff_name = "_local_staled_grad_cnt_abc_" + str(
                        self._iter_cnt % 2) + "_" + str(rank)

                    staled_idx_send_buff = self._create_shm_tensor(
                        staled_idx_send_buff_name, self._idx_dtype,
                        (part_size, ))
                    staled_grad_send_buff = self._create_shm_tensor(
                        staled_grad_send_buff_name, self._emb_dtype,
                        (part_size, self._emb_dim))
                    staled_grad_cnt_send_buff = self._create_shm_tensor(
                        staled_grad_cnt_send_buff_name, torch.int64,
                        (part_size, ))

                    staled_idx_send_buff.copy_(local_staled_idx[part_index])
                    staled_grad_send_buff.copy_(local_staled_grads[part_index])
                    staled_grad_cnt_send_buff.copy_(
                        local_staled_cnt[part_index])

                    self._send_shm_tensor_to_one(staled_idx_send_buff_name,
                                                 rank)
                    self._send_shm_tensor_to_one(staled_grad_send_buff_name,
                                                 rank)
                    self._send_shm_tensor_to_one(
                        staled_grad_cnt_send_buff_name, rank)
                else:
                    self._send_shm_tensor_to_one(EMPTY_TENSOR_NAME, rank)
                    self._send_shm_tensor_to_one(EMPTY_TENSOR_NAME, rank)
                    self._send_shm_tensor_to_one(EMPTY_TENSOR_NAME, rank)

        else:
            self._send_shm_tensor_to_all(EMPTY_TENSOR_NAME)
            self._send_shm_tensor_to_all(EMPTY_TENSOR_NAME)
            self._send_shm_tensor_to_all(EMPTY_TENSOR_NAME)

        # reset buffer
        self._local_grad_buff[local_staled_index][:] = 0
        self._local_grad_cnt_buff[local_staled_index] = 0

        # update local iter
        self._local_grad_iter_count[local_cnt_index] += 1
        self._local_grad_iter_count[local_staled_index] = 0

        toc = time.time()
        self._time_send_local_staled += toc - tic

    def _update_cache(self, ids_list, embs_list):
        # insert each client's emb one by one
        for i in range(self.num_clients):
            if ids_list[i].numel() <= 0:
                continue

            if ids_list[i].size(0) != embs_list[i].size(0):
                print("ids_list[i].size(0) != embs_list[i].size(0)")
                print(ids_list[i].shape)
                print(embs_list[i].shape)
                continue
                # assert False

            tic = time.time()
            locs = self._idx_cache.add(ids_list[i])
            toc = time.time()
            self._update_time_insert_cache[i] += toc - tic

            tic = time.time()

            insert_mask = locs >= 0
            valid_index = torch.nonzero(insert_mask).flatten()

            if valid_index.max() > embs_list[i].size(0):
                print("valid_index.max() > embs_list[i].size(0)")
                print(valid_index.max())
                print(embs_list[i].size(0))
                assert False

            valid_locs = locs[valid_index]
            # print(valid_locs.shape)
            # print(embs_list[i].shape)
            self._emb_cache_buff[valid_locs] = embs_list[i][valid_index]

            toc = time.time()
            self._update_time_insert_emb[i] += toc - tic

            tic = time.time()
            self._grad_cache_buff[valid_locs][:] = 0
            self._grad_cnt_cache_buff[valid_locs] = 0
            self._grad_idx_cache_buff[valid_locs] = -1
            toc = time.time()
            self._update_time_empty_grad[i] += toc - tic

            self._num_ids_uncached[i] += ids_list[i].numel()
            self._num_ids_real_add[i] += valid_locs.numel()

    def _accumulate_gradients_cold_remote(self, id, grad):
        # todo (ping): assume ids are unique
        tic = time.time()

        locs = self._idx_cache.lookup(id)
        cached_mask = locs >= 0
        cahce_idx = torch.nonzero(cached_mask).flatten()

        if cahce_idx.numel() > 0:
            _tmp = locs[cahce_idx]
            self._grad_idx_cache_buff[_tmp] = id[cahce_idx]
            self._grad_iter_threshold[_tmp] = self.threshold[id[cahce_idx]]
            self._grad_cache_buff[_tmp] += grad[cahce_idx]
            self._grad_cnt_cache_buff[_tmp] += 1

        toc = time.time()
        self._accum_time_cold_remote += toc - tic

    def _accumulate_gradients_cold_local(self, id, grad):
        # todo (ping): assume ids are unique
        tic = time.time()

        if id.numel() > 0:
            local_id = id - self._part_start
            self._local_grad_buff[local_id] += grad
            self._local_grad_cnt_buff[local_id] += 1

        toc = time.time()
        self._accum_time_cold_local += toc - tic

    def _timecheck(self):
        print("=============Server Time Log============")
        print("Iter {}".format(self._iter_cnt))
        for i in range(self.num_clients):
            print("For client {}".format(i))
            print("Insert idx: {:.3f} ms".format(
                self._update_time_insert_cache[i] / self._iter_cnt * 1000))
            print("Insert emb data: {:.3f} ms".format(
                self._update_time_insert_emb[i] / self._iter_cnt * 1000))
            print("Empty grad: {:.3f} ms".format(
                self._update_time_empty_grad[i] / self._iter_cnt * 1000))
            print("#Ids uncached: {:.3f}".format(self._num_ids_uncached[i] /
                                                 self._iter_cnt))
            print("#Ids real add: {:.3f}".format(self._num_ids_real_add[i] /
                                                 self._iter_cnt))
            print("#cold remote grads: {:.3f}".format(
                self._num_cold_remote_grads[i] / self._iter_cnt))
            print("#cold local grads: {:.3f}".format(
                self._num_cold_local_grads[i] / self._iter_cnt))
        print("Recv cold grads: {:.3f} ms".format(self._time_recv_cold_grads /
                                                  self._iter_cnt * 1000))
        print("Accum cold remote grads: {:.3f} ms".format(
            self._accum_time_cold_remote / self._iter_cnt * 1000))
        print("Accum cold local grads: {:.3f} ms".format(
            self._accum_time_cold_local / self._iter_cnt * 1000))
        print("Send remote staled grads: {:.3f} ms".format(
            self._time_send_remote_staled / self._iter_cnt * 1000))
        print("Send local staled grads: {:.3f} ms".format(
            self._time_send_local_staled / self._iter_cnt * 1000))
        print("========================================")
