import torch
from dgl.distributed.dist_tensor import DistTensor
import abc
from abc import abstractmethod
from .common import *
from .utils import *
import time
import EmbCacheLib
import torch.distributed as dist
from .cache import LocalSparseAdamStateCache
import dgl


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


EMB_STATES = "emb_states"
WORLD_SIZE = "world_size"
IDS = "ids"
PARAMS = "params"
STATES = "states"


class DistSparseGradOptimizer(abc.ABC):
    r"""The abstract dist sparse optimizer.

    Note: dgl dist sparse optimizer only work with dgl.distributed.DistEmbedding

    Parameters
    ----------
    params : list of DistEmbedding
        The list of DistEmbedding.
    lr : float
        The learning rate.
    """

    def __init__(self, params, lr, part_book, local_group, partition_group,
                 hotness, hot_threshold):
        self._params = params
        self._lr = lr
        self._rank = None
        self._world_size = None
        self._shared_cache = {}
        self._clean_grad = False
        self._opt_meta = {}
        self._state = {}
        ## collect all hyper parameters for save
        self._defaults = {}

        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        self._num_parts = part_book.num_partitions()
        self._num_trainers = self._world_size // self._num_parts
        self._partition_range = [0]
        for i in range(part_book.num_partitions()):
            self._partition_range.append(part_book._max_node_ids[i])
        self._partition_range = torch.tensor(self._partition_range).cuda()

        self._local_group = local_group
        self._partition_group = partition_group
        self._hot_mask = (hotness >= hot_threshold).cuda()

    def step(self):
        """The step function.

        The step function is invoked at the end of every batch to push the gradients
        of the embeddings involved in a mini-batch to DGL's servers and update the embeddings.
        """
        with torch.no_grad():
            local_indics = {emb.name: [] for emb in self._params}
            local_grads = {emb.name: [] for emb in self._params}
            local_cnts = {emb.name: [] for emb in self._params}
            device = torch.device("cpu")
            for emb in self._params:
                name = emb.weight.name
                kvstore = emb.weight.kvstore

                idics = []
                grads = []
                for trace in emb._trace:
                    if trace[1].grad is not None:
                        idics.append(trace[0])
                        grads.append(trace[1].grad.data)
                    else:
                        assert len(trace[0]) == 0

                idics = (torch.cat(
                    idics, dim=0) if len(idics) != 0 else torch.zeros(
                        (0, ), dtype=torch.long, device=torch.device("cpu")))
                grads = (torch.cat(grads, dim=0)
                         if len(grads) != 0 else torch.zeros(
                             (0, emb.embedding_dim),
                             dtype=torch.float32,
                             device=torch.device("cpu"),
                         ))
                device = grads.device

                # split hot & cold
                batch_hot_mask = self._hot_mask[idics]
                hot_idics = idics[batch_hot_mask]
                hot_grads = grads[batch_hot_mask]
                cold_idics = idics[~batch_hot_mask]
                cold_grads = grads[~batch_hot_mask]

                # hot local aggregation
                if self._num_trainers > 1:
                    (
                        hot_idics,
                        hot_grads,
                        _,
                        split_time,
                        size_all2all_time,
                        idx_all2all_time,
                        grad_all2all_time,
                        _,
                        hot_grads_handlers,
                    ) = swap_gradients(
                        hot_idics,
                        hot_grads,
                        None,
                        1,
                        self._num_trainers,
                        device,
                        group=self._local_group,
                    )
                    wait_handlers(hot_grads_handlers)
                hot_idics, hot_grads, hot_cnts = only_unique_grads(
                    hot_idics, hot_grads)
                idics = torch.cat([hot_idics, cold_idics])
                grads = torch.cat([hot_grads, cold_grads])
                cnts = torch.cat([
                    hot_cnts,
                    torch.ones_like(cold_idics,
                                    dtype=torch.int64,
                                    device="cuda")
                ])

                # will send grad to each corresponding trainer
                if self._world_size > 1:
                    (
                        local_indics[name],
                        local_grads[name],
                        local_cnts[name],
                        split_time,
                        size_all2all_time,
                        idx_all2all_time,
                        grad_all2all_time,
                        cnt_all2all_time,
                        hot_grads_handlers,
                    ) = swap_gradients(
                        idics,
                        grads,
                        self._partition_range,
                        self._num_parts,
                        1,
                        device,
                        grads_cnt=cnts,
                        group=self._partition_group,
                    )

                else:
                    local_indics[name] = idics
                    local_grads[name] = grads
                    local_cnts[name] = cnts

            if self._clean_grad:
                # clean gradient track
                for emb in self._params:
                    emb.reset_trace()
                self._clean_grad = False

            if self._world_size > 1:
                wait_handlers(hot_grads_handlers)

            # do local update
            for emb in self._params:
                name = emb.weight.name
                idx = local_indics[name]
                grad = local_grads[name]
                cnt = local_cnts[name]
                self.update(
                    idx.to(device, non_blocking=True),
                    grad.to(device, non_blocking=True),
                    emb,
                    cnt=cnt,
                    is_unique=False,
                )

        # synchronized gradient update
        if self._world_size > 1:
            torch.distributed.barrier()

    @abstractmethod
    def update(self, idx, grad, emb):
        """Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. We maintain gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.distributed.DistEmbedding
            Sparse node embedding to update.
        """

    def zero_grad(self):
        """clean grad cache"""
        self._clean_grad = True


def initializer(shape, dtype):
    """Sparse optimizer state initializer

    Parameters
    ----------
    shape : tuple of ints
        The shape of the state tensor
    dtype : torch dtype
        The data type of the state tensor
    """
    arr = torch.zeros(shape, dtype=dtype)
    return arr


class SparseAdam(DistSparseGradOptimizer):
    r"""Distributed Node embedding optimizer using the Adam algorithm.

    This optimizer implements a distributed sparse version of Adam algorithm for
    optimizing :class:`dgl.distributed.DistEmbedding`. Being sparse means it only updates
    the embeddings whose gradients have updates, which are usually a very
    small portion of the total embeddings.

    Adam maintains a :math:`Gm_{t,i,j}` and `Gp_{t,i,j}` for every parameter
    in the embeddings, where
    :math:`Gm_{t,i,j}=beta1 * Gm_{t-1,i,j} + (1-beta1) * g_{t,i,j}`,
    :math:`Gp_{t,i,j}=beta2 * Gp_{t-1,i,j} + (1-beta2) * g_{t,i,j}^2`,
    :math:`g_{t,i,j} = lr * Gm_{t,i,j} / (1 - beta1^t) / \sqrt{Gp_{t,i,j} / (1 - beta2^t)}` and
    :math:`g_{t,i,j}` is the gradient of the dimension :math:`j` of embedding :math:`i`
    at step :math:`t`.

    NOTE: The support of sparse Adam optimizer is experimental.

    Parameters
    ----------
    params : list[dgl.distributed.DistEmbedding]
        The list of dgl.distributed.DistEmbedding.
    lr : float
        The learning rate.
    betas : tuple[float, float], Optional
        Coefficients used for computing running averages of gradient and its square.
        Default: (0.9, 0.999)
    eps : float, Optional
        The term added to the denominator to improve numerical stability
        Default: 1e-8
    """

    def __init__(self,
                 params,
                 lr,
                 part_book,
                 local_group,
                 partition_group,
                 hotness,
                 hot_threshold,
                 betas=(0.9, 0.999),
                 eps=1e-08):
        super(SparseAdam,
              self).__init__(params, lr, part_book, local_group,
                             partition_group, hotness, hot_threshold)
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
            state = (state_step, state_mem, state_power)
            assert (emb.name not in self._state
                    ), "{} already registered in the optimizer".format(
                        emb.name)
            self._state[emb.name] = state

    def update(self, idx, grad, emb, cnt=None, is_unique=False):
        beta1 = self._beta1
        beta2 = self._beta2
        eps = self._eps
        clr = self._lr
        state_step, state_mem, state_power = self._state[emb.name]

        state_dev = torch.device("cpu")
        exec_dev = grad.device

        # only perform async copies cpu -> gpu, or gpu-> gpu, but block
        # when copying to the cpu, so as to ensure the copy is finished
        # before operating on the data on the cpu
        state_block = state_dev == torch.device(
            "cpu") and exec_dev != state_dev

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
        state_idx = grad_indices.to(state_dev)
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
        state_val = state_step[state_idx] + 1
        state_step[state_idx] = state_val
        state_step = state_val.to(exec_dev)
        orig_mem = state_mem[state_idx].to(exec_dev)
        orig_power = state_power[state_idx].to(exec_dev)

        grad_mem = grad_values
        grad_power = grad_values * grad_values
        update_mem = beta1 * orig_mem + (1.0 - beta1) * grad_mem
        update_power = beta2 * orig_power + (1.0 - beta2) * grad_power
        update_mem_dst = update_mem.to(state_dev, non_blocking=False)
        update_power_dst = update_power.to(state_dev, non_blocking=False)
        if state_block:
            # use events to try and overlap CPU and GPU as much as possible
            update_event = torch.cuda.Event()
            update_event.record()

        update_mem_corr = update_mem / (1.0 - torch.pow(
            torch.tensor(beta1, device=exec_dev), state_step)).unsqueeze(1)
        update_power_corr = update_power / (1.0 - torch.pow(
            torch.tensor(beta2, device=exec_dev), state_step)).unsqueeze(1)
        std_values = clr * update_mem_corr / (torch.sqrt(update_power_corr) +
                                              eps)

        std_values_dst = std_values.to(state_dev, non_blocking=False)

        if state_block:
            std_event = torch.cuda.Event()
            std_event.record()
            # wait for our transfers from exec_dev to state_dev to finish
            # before we can use them
            update_event.wait()
        state_mem[state_idx] = update_mem_dst
        state_power[state_idx] = update_power_dst

        if state_block:
            # wait for the transfer of std_values to finish before we
            # can use it
            std_event.wait()
        # torch.cuda.synchronize()
        update_grad_compute_time = time.time() - tic

        tic = time.time()
        emb._tensor[state_idx] -= std_values_dst
        # torch.cuda.synchronize()
        update_emb_value_time = time.time() - tic

        return unique_time, update_grad_compute_time, update_emb_value_time, update_num_unique
