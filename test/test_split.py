import EmbCacheLib
import torch
import time


torch.manual_seed(42)
LEN = 2000_000

_remote_mask = torch.zeros(LEN).bool()
remote_idx = torch.randint(0, LEN, (LEN // 2, ))
_remote_mask[remote_idx] = True


_hot_mask = torch.zeros(LEN).bool()
hot_idx = torch.randint(0, LEN, (LEN // 2, ))
_hot_mask[hot_idx] = True

_total_mask = torch.zeros(LEN).char()
_total_mask[_hot_mask & _remote_mask] = 1
_total_mask[(~_hot_mask) & _remote_mask] = 2
_total_mask = _total_mask.cuda()

# print(_total_mask)

for i in range(10):
    
    index = torch.randint(0, LEN, (1000_000, ))
    begin = time.time()
    # split hot/cold
    hot_mask = _hot_mask[index]
    idx_hot = index[hot_mask]
    idx_cold = index[~hot_mask]
    
    # split local/remote
    remote_cold_mask = _remote_mask[idx_cold]
    idx_remote_cold = idx_cold[remote_cold_mask]
    idx_local_cold = idx_cold[~remote_cold_mask]
    end = time.time()
    
    print((end - begin) * 1000)

print()

for i in range(10):
    
    index = torch.randint(0, LEN, (200_000, )).cuda()
    
    begin = time.time()
    EmbCacheLib.cudaSplit(index, _total_mask)
    end = time.time()
    
    print((end - begin) * 1000)
    