import torch
from dgl.distributed.dist_tensor import DistTensor
from abc import ABC
from abc import abstractmethod
from .common import *
from .utils import *
import time
import EmbCacheLib
import torch.distributed as dist
from .cache import LocalSparseAdamStateCache


def wait_handlers(handlers):
    for handler in handlers:
        if handler is not None:
            handler.wait()


def swap_gradients(
    idics,
    grads,
    partition_range,
    num_parts,
    num_trainers_per_part,
    device,
    grads_cnt=None,
    group=None,
):
    split_time = 0
    size_all2all_time = 0
    idx_all2all_time = 0
    grad_all2all_time = 0
    cnt_all2all_time = 0

    tic = time.time()
    if idics.numel() > 0:
        if partition_range is None:
            split = EmbCacheLib.cudaReminder(idics, num_trainers_per_part)
        else:
            split = EmbCacheLib.cudaBinarySearch(idics, partition_range,
                                                 num_trainers_per_part)

        value, indices = torch.sort(split)
        ids = idics[indices]
        grads = grads[indices]
        if grads_cnt is not None:
            grads_cnt = grads_cnt[indices]

        # torch.cuda.synchronize()
        gpu_ranges = torch.arange(int(num_parts * num_trainers_per_part),
                                  device='cuda')
        input_splits_tensor = EmbCacheLib.cudaBinarySearch2(gpu_ranges, value)
        input_splits_tensor[
            1:] = input_splits_tensor[1:] - input_splits_tensor[:-1]
        input_splits_sizes = input_splits_tensor.tolist()

    else:
        ids = idics
        grads = grads
        if grads_cnt is not None:
            grads_cnt = grads_cnt
        input_splits_tensor = torch.zeros(
            (num_parts * num_trainers_per_part, ),
            dtype=torch.int64,
            device=device)
        input_splits_sizes = [0] * (num_parts * num_trainers_per_part)

    # torch.cuda.synchronize()
    split_time += time.time() - tic

    tic = time.time()
    output_splits_tensor = torch.empty_like(input_splits_tensor)
    dist.all_to_all_single(output_splits_tensor,
                           input_splits_tensor,
                           group=group)
    # torch.cuda.synchronize()
    output_splits_sizes = output_splits_tensor.tolist()
    num_emb = sum(output_splits_sizes)
    size_all2all_time += time.time() - tic

    tic = time.time()
    output_ids = torch.empty((num_emb, ), dtype=ids.dtype, device=device)
    ids_handler = dist.all_to_all_single(
        output_ids,
        ids,
        output_split_sizes=output_splits_sizes,
        input_split_sizes=input_splits_sizes,
        group=group,
        async_op=True)
    # torch.cuda.synchronize()
    idx_all2all_time += time.time() - tic

    tic = time.time()
    output_grads = torch.empty((num_emb, grads.shape[1]),
                               dtype=grads.dtype,
                               device=device)
    grads_handler = dist.all_to_all_single(
        output_grads,
        grads,
        output_split_sizes=output_splits_sizes,
        input_split_sizes=input_splits_sizes,
        group=group,
        async_op=True)
    # torch.cuda.synchronize()
    grad_all2all_time += time.time() - tic

    if grads_cnt is not None:
        tic = time.time()
        output_grads_cnt = torch.empty((num_emb, ),
                                       dtype=grads_cnt.dtype,
                                       device=device)
        cnt_handler = dist.all_to_all_single(
            output_grads_cnt,
            grads_cnt,
            output_split_sizes=output_splits_sizes,
            input_split_sizes=input_splits_sizes,
            group=group,
            async_op=True)
        # torch.cuda.synchronize()
        cnt_all2all_time += time.time() - tic

    if grads_cnt is not None:
        return (output_ids, output_grads, output_grads_cnt, split_time,
                size_all2all_time, idx_all2all_time, grad_all2all_time,
                cnt_all2all_time, (ids_handler, grads_handler, cnt_handler))
    else:
        return (output_ids, output_grads, None, split_time, size_all2all_time,
                idx_all2all_time, grad_all2all_time, cnt_all2all_time,
                (ids_handler, grads_handler))


def only_unique_grads(idx, grad):
    grad_indices, inverse, grad_cnt = torch.unique(idx,
                                                   return_inverse=True,
                                                   return_counts=True)
    grad_values = torch.zeros((grad_indices.shape[0], grad.shape[1]),
                              device=grad.device)
    grad_values.index_add_(0, inverse, grad)
    return grad_indices, grad_values, grad_cnt


def unique_grads(idx, grad, cnt=None):
    if cnt is not None:
        grad_indices, inverse = torch.unique(idx, return_inverse=True)
        grad_cnt = torch.ones((grad_indices.shape[0], ),
                              dtype=cnt.dtype,
                              device=cnt.device)
        grad_cnt.index_add_(0, inverse, cnt)
    else:
        grad_indices, inverse, grad_cnt = torch.unique(idx,
                                                       return_inverse=True,
                                                       return_counts=True)
    grad_values = torch.zeros((grad_indices.shape[0], grad.shape[1]),
                              device=grad.device)
    grad_values.index_add_(0, inverse, grad)
    grad_values = grad_values / grad_cnt.unsqueeze(1)
    return grad_indices, grad_values


class DistembSparseGradOptimizer(ABC):

    def __init__(self, params, lr, num_parts, local_group, partition_group):
        self._params = params
        self._lr = lr

        self._clean_grad = False

        self._state = {}
        self._defaults = {}

        self._rank = None
        self._world_size = None
        self._num_parts = num_parts
        self._local_group = local_group
        self._partition_group = partition_group

        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        self._reset_recorder()

    def _reset_recorder(self):
        self._time_split_grad = 0
        self._time_get_staled = 0
        self._time_send_cold_grad = 0
        self._iter_cnt = 0
        self._time_swap_hot = [0, 0, 0, 0]
        self._time_swap_cold_remote = [0, 0, 0, 0, 0]
        self._time_update_hot_unique = 0
        self._time_update_hot_compute = 0
        self._time_update_hot_update_emb = 0
        self._time_update_cold_remote = 0
        self._time_update_cold_local = 0
        self._time_hot_local_swap = [0, 0, 0, 0]
        self._time_hot_local_unique = 0

        self._num_idx = 0
        self._num_idx_cold_local = 0
        self._num_idx_cold_remote = 0
        self._num_idx_hot = 0
        self._num_idx_hot_local_unique = 0
        self._num_idx_staled_cold_remote = 0
        self._num_idx_staled_cold_local = 0

        self._num_hot_update_idx = 0
        self._num_hot_update_idx_unique = 0

    def step(self):
        with torch.no_grad():
            local_hot_indics = {}
            local_hot_grads = {}
            local_hot_cnts = {}
            for emb in self._params:
                name = emb.name
                enable_cache = emb.enable_cache
                trainers_per_server = self._world_size // self._num_parts
                idics = []
                grads = []
                trace = emb.trace

                idics = [t[0] for t in trace]
                grads = [t[1].grad.data for t in trace]
                if enable_cache:
                    hot_mask = [t[2] for t in trace]
                    local_mask = [t[3] for t in trace]
                device = grads[0].device

                # initialize idx and grads if empty
                idics = (torch.cat(idics, dim=0) if len(idics) != 0 else
                         torch.zeros((0, ), dtype=torch.long, device=device))
                grads = (torch.cat(grads, dim=0) if len(grads) != 0 else
                         torch.zeros((0, emb.embedding_dim),
                                     dtype=torch.float32,
                                     device=device))
                self._num_idx += idics.shape[0]

                if enable_cache:
                    tic = time.time()
                    hot_mask = (torch.cat(hot_mask, dim=0)
                                if len(hot_mask) != 0 else torch.zeros(
                                    (0, ), dtype=torch.bool, device=device))
                    local_mask = (torch.cat(local_mask, dim=0)
                                  if len(local_mask) != 0 else torch.zeros(
                                      (0, ), dtype=torch.bool, device=device))
                    local_cold_mask = ~(hot_mask) & local_mask
                    remote_cold_mask = ~(hot_mask) & ~(local_mask)

                    idics_cold_local = idics[local_cold_mask]
                    grads_cold_local = grads[local_cold_mask]

                    idics_cold_remote = idics[remote_cold_mask]
                    grads_cold_remote = grads[remote_cold_mask]

                    idics = idics[hot_mask]
                    grads = grads[hot_mask]
                    # torch.cuda.synchronize()
                    toc = time.time()
                    self._time_split_grad += toc - tic
                    self._num_idx_cold_local += idics_cold_local.shape[0]
                    self._num_idx_cold_remote += idics_cold_remote.shape[0]
                    self._num_idx_hot += idics.shape[0]

                if trainers_per_server > 1:
                    # local all2all
                    (
                        idics,
                        grads,
                        _,
                        split_time,
                        size_all2all_time,
                        idx_all2all_time,
                        grad_all2all_time,
                        _,
                        hot_grads_handlers,
                    ) = swap_gradients(
                        idics,
                        grads,
                        None,
                        1,
                        trainers_per_server,
                        device,
                        group=self._local_group,
                    )
                    self._time_hot_local_swap[0] += split_time
                    self._time_hot_local_swap[1] += size_all2all_time
                    self._time_hot_local_swap[2] += idx_all2all_time
                    self._time_hot_local_swap[3] += grad_all2all_time

                # merge hot grads and remote cold grads
                if enable_cache:
                    tic = time.time()
                    staled_remote_cold, staled_local_cold = emb.client.get_staled_gradients(
                        device)
                    staled_remote_cold[3].synchronize()
                    staled_remote_cold_idx = staled_remote_cold[0]
                    staled_remote_cold_grads = staled_remote_cold[1]
                    staled_remote_cold_cnt = staled_remote_cold[2]
                    # torch.cuda.synchronize()
                    toc = time.time()
                    self._time_get_staled += toc - tic
                    self._num_idx_staled_cold_remote += staled_remote_cold_idx.shape[
                        0]

                    tic = time.time()
                    wait_handlers(hot_grads_handlers)
                    idics, grads, cnts = only_unique_grads(idics, grads)
                    # torch.cuda.synchronize()
                    self._time_hot_local_unique += time.time() - tic
                    self._num_idx_hot_local_unique += idics.shape[0]

                    if staled_remote_cold_idx.numel() > 0:
                        idics = torch.cat([idics, staled_remote_cold_idx])
                        grads = torch.cat([grads, staled_remote_cold_grads])
                        cnts = torch.cat([cnts, staled_remote_cold_cnt])

                # swap hot grads between partitions
                if self._num_parts > 1:
                    (
                        local_hot_indics[name],
                        local_hot_grads[name],
                        local_hot_cnts[name],
                        split_time,
                        size_all2all_time,
                        idx_all2all_time,
                        grad_all2all_time,
                        cnt_all2all_time,
                        hot_grads_handlers,
                    ) = swap_gradients(
                        idics,
                        grads,
                        emb._partition_range,
                        self._num_parts,
                        1,
                        device,
                        grads_cnt=cnts,
                        group=self._partition_group,
                    )
                    self._time_swap_hot[0] += split_time
                    self._time_swap_hot[1] += size_all2all_time
                    self._time_swap_hot[2] += idx_all2all_time
                    self._time_swap_hot[3] += grad_all2all_time
                else:
                    local_hot_indics[name] = idics
                    local_hot_grads[name] = grads
                    local_hot_cnts[name] = cnts
                    hot_grads_handlers = (None, None)

                if enable_cache:
                    tic = time.time()
                    # create and try send cold gradients
                    emb.client.create_send_cold_gradients_buffer(
                        idics_cold_remote,
                        grads_cold_remote,
                        idics_cold_local,
                        grads_cold_local,
                    )
                    # torch.cuda.synchronize()
                    toc = time.time()
                    self._time_send_cold_grad += toc - tic

            if self._clean_grad:
                # clean gradient track
                for emb in self._params:
                    emb.reset_trace()
                self._clean_grad = False

            # do hot grads local update
            for emb in self._params:
                name = emb.name
                wait_handlers(hot_grads_handlers)
                idx = local_hot_indics[name]
                self._num_hot_update_idx += idx.numel()
                if idx.numel() > 0:
                    grad = local_hot_grads[name]
                    cnt = local_hot_cnts[name]
                    (
                        unique_time,
                        compute_time,
                        update_emb_value_time,
                        unique_update_idx_num,
                    ) = self.update(
                        idx.to(device, non_blocking=False),
                        grad.to(device, non_blocking=False),
                        emb,
                        cnt=cnt,
                        is_unique=False,
                    )
                    self._time_update_hot_unique += unique_time
                    self._time_update_hot_compute += compute_time
                    self._time_update_hot_update_emb += update_emb_value_time
                    self._num_hot_update_idx_unique += unique_update_idx_num

                if emb.enable_cache:
                    # try send cold gradients
                    tic = time.time()
                    emb.client.try_send_cold_grads()
                    # torch.cuda.synchronize()
                    toc = time.time()
                    self._time_send_cold_grad += toc - tic

                    staled_local_cold[3].synchronize()
                    staled_local_cold_idx = staled_local_cold[0]
                    staled_local_cold_grads = staled_local_cold[1]
                    staled_local_cold_cnt = staled_local_cold[2]
                    self._num_idx_staled_cold_local += staled_local_cold_idx.shape[
                        0]

                    # local cold update
                    tic = time.time()
                    if staled_local_cold_idx.numel() > 0:
                        self.update(
                            staled_local_cold_idx,
                            staled_local_cold_grads,
                            emb,
                            cnt=staled_local_cold_cnt,
                            is_unique=True,
                        )
                    # torch.cuda.synchronize()
                    toc = time.time()
                    self._time_update_cold_local += toc - tic

                    # try send cold gradients
                    tic = time.time()
                    emb.client.try_send_cold_grads()
                    # torch.cuda.synchronize()
                    toc = time.time()
                    self._time_send_cold_grad += toc - tic

                    # try send cold gradients
                    tic = time.time()
                    emb.client.force_send_cold_grads()
                    # torch.cuda.synchronize()
                    toc = time.time()
                    self._time_send_cold_grad += toc - tic

        # synchronized gradient update
        if self._world_size > 1:
            torch.distributed.barrier()

        self._iter_cnt += 1
        if self._iter_cnt % 10 == 0:
            pass
            # self.timecheck()
            # self._reset_recorder()

        # if not self._state_cache_built:
        #     self._build_state_cache()
        # else:
        #     self._print_hit_rate()

    @abstractmethod
    def update(self, idx, grad, emb, cnt=None, is_unique=False):
        pass

    @abstractmethod
    def _build_state_cache(self, reserved_mem=1.0):
        # reserved_mem unit: GB
        pass

    @abstractmethod
    def _print_hit_rate(self):
        pass

    def zero_grad(self):
        """clean grad cache"""
        self._clean_grad = True

    def timecheck(self):
        timetable = (
            "==========SparseOptim Time Log==========\n"
            "Iter {}\n"
            "Split grad: {:.3f} ms\n"
            "Get staled grad: {:.3f} ms\n"
            "Send cold grad: {:.3f} ms\n"
            "hot grad local swap (split,size,idx,grad): {:.3f},{:.3f},{:.3f},{:.3f} ms\n"
            "hot grad local unique: {:.3f} ms\n"
            "Swap hot grad (split,size,idx,grad): {:.3f},{:.3f},{:.3f},{:.3f} ms\n"
            "Swap cold remote grad (split,size,idx,grad,cnt): {:.3f},{:.3f},{:.3f},{:.3f},{:.3f} ms\n"
            "Update hot grad - unique: {:.3f} ms\n"
            "Update hot grad - compute: {:.3f} ms\n"
            "Update hot grad - emb update: {:.3f} ms\n"
            "Update cold remote grad: {:.3f} ms\n"
            "Update cold local grad: {:.3f} ms\n"
            "#idx: {:.3f}\n"
            "#idx_hot: {:.3f}\n"
            "#idx_hot_local_unqiue: {:.3f}\n"
            "#idx_cold_remote: {:.3f}\n"
            "#idx_cold_local: {:.3f}\n"
            "#idx_staled_cold_remote: {:.3f}\n"
            "#idx_staled_cold_loccal: {:.3f}\n"
            "#idx_hot_update: {:.3f}\n"
            "#idx_hot_update_unique: {:.3f}\n"
            "========================================\n".format(
                self._iter_cnt, self._time_split_grad / self._iter_cnt * 1000,
                self._time_get_staled / self._iter_cnt * 1000,
                self._time_send_cold_grad / self._iter_cnt * 1000,
                self._time_hot_local_swap[0] / self._iter_cnt * 1000,
                self._time_hot_local_swap[1] / self._iter_cnt * 1000,
                self._time_hot_local_swap[2] / self._iter_cnt * 1000,
                self._time_hot_local_swap[3] / self._iter_cnt * 1000,
                self._time_hot_local_unique / self._iter_cnt * 1000,
                self._time_swap_hot[0] / self._iter_cnt * 1000,
                self._time_swap_hot[1] / self._iter_cnt * 1000,
                self._time_swap_hot[2] / self._iter_cnt * 1000,
                self._time_swap_hot[3] / self._iter_cnt * 1000,
                self._time_swap_cold_remote[0] / self._iter_cnt * 1000,
                self._time_swap_cold_remote[1] / self._iter_cnt * 1000,
                self._time_swap_cold_remote[2] / self._iter_cnt * 1000,
                self._time_swap_cold_remote[3] / self._iter_cnt * 1000,
                self._time_swap_cold_remote[4] / self._iter_cnt * 1000,
                self._time_update_hot_unique / self._iter_cnt * 1000,
                self._time_update_hot_compute / self._iter_cnt * 1000,
                self._time_update_hot_update_emb / self._iter_cnt * 1000,
                self._time_update_cold_remote / self._iter_cnt * 1000,
                self._time_update_cold_local / self._iter_cnt * 1000,
                self._num_idx / self._iter_cnt, self._num_idx_hot /
                self._iter_cnt, self._num_idx_hot_local_unique /
                self._iter_cnt, self._num_idx_cold_remote / self._iter_cnt,
                self._num_idx_cold_local / self._iter_cnt,
                self._num_idx_staled_cold_remote / self._iter_cnt,
                self._num_idx_staled_cold_local / self._iter_cnt,
                self._num_hot_update_idx / self._iter_cnt,
                self._num_hot_update_idx_unique / self._iter_cnt))
        print(timetable)


class SparseAdam(DistembSparseGradOptimizer):

    def __init__(
            self,
            params,
            lr,
            num_parts,
            local_group,
            partition_group,
            betas=(0.9, 0.999),
            eps=1e-08,
    ):
        super(SparseAdam, self).__init__(params, lr, num_parts, local_group,
                                         partition_group)
        self._eps = eps
        # We need to register a state sum for each embedding in the kvstore.
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._defaults = {
            "_lr": lr,
            "_eps": eps,
            "_beta1": betas[0],
            "_beta2": betas[1],
        }
        self._state = {}

        for emb in params:
            state_step = DistTensor(
                (emb.num_embeddings, ),
                torch.float32,
                emb.name + "_step",
                init_func=zero_initializer,
                is_gdata=False,
            )
            state_mem = DistTensor(
                (emb.num_embeddings, emb.embedding_dim),
                torch.float32,
                emb.name + "_mem",
                init_func=zero_initializer,
                is_gdata=False,
            )
            state_power = DistTensor(
                (emb.num_embeddings, emb.embedding_dim),
                torch.float32,
                emb.name + "_power",
                init_func=zero_initializer,
                is_gdata=False,
            )
            assert (emb.name not in self._state
                    ), "{} already registered in the optimizer".format(
                        emb.name)
            self._state[emb.name] = LocalSparseAdamStateCache(emb,
                                                              state_step,
                                                              state_mem,
                                                              state_power,
                                                              count_hit=False)

        self._state_cache_built = False

    def update(self, idx, grad, emb, cnt=None, is_unique=False):
        beta1 = self._beta1
        beta2 = self._beta2
        eps = self._eps
        clr = self._lr
        state = self._state[emb.name]

        # state_dev = torch.device("cpu")
        # exec_dev = grad.device

        # only perform async copies cpu -> gpu, or gpu-> gpu, but block
        # when copying to the cpu, so as to ensure the copy is finished
        # before operating on the data on the cpu
        # state_block = state_dev == torch.device(
        #    "cpu") and exec_dev != state_dev

        tic = time.time()
        if is_unique:
            grad_indices = idx
            if cnt is not None:
                grad_values = grad / cnt.unsqueeze(1)
            else:
                grad_values = grad
        else:
            grad_indices, grad_values = unique_grads(idx, grad, cnt)
        # torch.cuda.synchronize()
        unique_time = time.time() - tic
        update_num_unique = grad_indices.numel()

        # update grad state
        tic = time.time()
        state_idx = grad_indices
        # The original implementation will cause read/write contension.
        #    state_step[state_idx] += 1
        #    state_step = state_step[state_idx].to(exec_dev, non_blocking=False)
        # In a distributed environment, the first line of code will send write requests to
        # kvstore servers to update the state_step which is asynchronous and the second line
        # of code will also send read requests to kvstore servers. The write and read requests
        # may be handled by different kvstore servers managing the same portion of the
        # state_step dist tensor in the same node. So that, the read request may read an old
        # value (i.e., 0 in the first iteration) which will cause
        # update_power_corr to be NaN
        state_step, orig_mem, orig_power = state[state_idx]

        state_step = state_step + 1
        # state_step = state_step.to(exec_dev)
        # orig_mem = orig_mem.to(exec_dev)
        # orig_power = orig_power.to(exec_dev)

        grad_mem = grad_values
        grad_power = grad_values * grad_values
        update_mem = beta1 * orig_mem + (1.0 - beta1) * grad_mem
        update_power = beta2 * orig_power + (1.0 - beta2) * grad_power

        state_step_dst = state_step  #.to(state_dev, non_blocking=False)
        update_mem_dst = update_mem  #.to(state_dev, non_blocking=False)
        update_power_dst = update_power  #.to(state_dev, non_blocking=False)
        # if state_block:
        #    # use events to try and overlap CPU and GPU as much as possible
        #    update_event = torch.cuda.Event()
        #    update_event.record()

        update_mem_corr = update_mem / (1.0 - torch.pow(
            torch.tensor(beta1, device='cuda'), state_step)).unsqueeze(1)
        update_power_corr = update_power / (1.0 - torch.pow(
            torch.tensor(beta2, device='cuda'), state_step)).unsqueeze(1)
        std_values = clr * update_mem_corr / (torch.sqrt(update_power_corr) +
                                              eps)

        std_values_dst = std_values.to('cpu', non_blocking=False)
        std_event = torch.cuda.Event()
        std_event.record()

        #if state_block:
        #    std_event = torch.cuda.Event()
        #    std_event.record()
        #    # wait for our transfers from exec_dev to state_dev to finish
        #    # before we can use them
        #    update_event.wait()

        state[state_idx] = state_step_dst, update_mem_dst, update_power_dst

        #if state_block:
        # wait for the transfer of std_values to finish before we
        # can use it
        #std_event.wait()
        # torch.cuda.synchronize()
        update_grad_compute_time = time.time() - tic

        tic = time.time()
        std_event.wait()
        emb.tensor[state_idx] -= std_values_dst
        # torch.cuda.synchronize()
        update_emb_value_time = time.time() - tic

        return unique_time, update_grad_compute_time, update_emb_value_time, update_num_unique

    def _build_state_cache(self, reserved_mem=1.0):
        gpu_capacity = torch.cuda.mem_get_info(
            torch.cuda.current_device())[1] - torch.cuda.max_memory_allocated(
            ) - reserved_mem * 1024 * 1024 * 1024

        # can't support heterogeneous graph now
        for key, value in self._state.items():
            value.create_cache(gpu_capacity)
            self._state[key] = value

        self._state_cache_built = True

    def _print_hit_rate(self):
        for key, value in self._state.items():
            print("{} hit rate {:.3f}".format(self._rank,
                                              value.get_hit_rate()))
