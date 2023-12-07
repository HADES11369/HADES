import torch


def create_cache_idx_map(num_items, cached_idx: torch.Tensor):
    idx_map = torch.full((num_items, ), -1)
    cached_num = cached_idx.shape[0]
    idx_map[cached_idx] = torch.arange(0, cached_num)
    idx_map = idx_map.to(torch.device("cuda"))
    return idx_map


class CachedTensor():

    def __init__(self, cpu_tensor: torch.Tensor, idx_map: torch.Tensor):
        assert len(cpu_tensor.shape) == 1 or len(cpu_tensor.shape) == 2
        assert idx_map.shape[0] == cpu_tensor.shape[0]

        cached_idx = (idx_map
                      != -1).nonzero().flatten().to(torch.device("cpu"))

        assert torch.max(cached_idx).item() < cpu_tensor.shape[0]
        assert torch.min(cached_idx).item() >= 0

        self._idx_map = idx_map
        self._cpu_tensor = cpu_tensor
        self._gpu_tensor = cpu_tensor[cached_idx].to(torch.device("cuda"))
        self._dim = cpu_tensor.shape[1] if len(cpu_tensor.shape) == 2 else 0

        self._hit_num = 0
        self._lookup_num = 0

    def index_fetch(self, idx):
        lookup_idx = self._idx_map[idx]
        cached_mask = lookup_idx != -1

        if self._dim == 0:
            buff = torch.empty((idx.shape[0], ), device=torch.device("cuda"))
        else:
            buff = torch.empty((idx.shape[0], self._dim),
                               device=torch.device("cuda"))

        idx_in_gpu = lookup_idx[cached_mask]
        buff[cached_mask] = self._gpu_tensor[idx_in_gpu]

        idx_in_cpu = idx[~cached_mask].to(torch.device("cpu"))
        buff[~cached_mask] = self._cpu_tensor[idx_in_cpu].to(
            torch.device("cuda"))

        self._hit_num += idx_in_gpu.shape[0]
        self._lookup_num += idx.shape[0]

        return buff

    def index_put(self, idx, val):
        lookup_idx = self._idx_map[idx]
        cached_mask = lookup_idx != -1

        idx_in_gpu = lookup_idx[cached_mask]
        val_in_gpu = val[cached_mask]

        idx_in_cpu = idx[~cached_mask].to(torch.device("cpu"))
        val_in_cpu = val[~cached_mask].to(torch.device("cpu"))

        self._gpu_tensor[idx_in_gpu] = val_in_gpu
        self._cpu_tensor[idx_in_cpu] = val_in_cpu

        self._hit_num += idx_in_gpu.shape[0]
        self._lookup_num += idx.shape[0]

    def get_hit_rate(self):
        return self._hit_num / self._lookup_num
