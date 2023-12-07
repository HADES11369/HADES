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
# _total_mask = _total_mask.cuda()


index = torch.randint(0, LEN, (200_000, )).cuda()
EmbCacheLib.cudaPin(_total_mask)


    
begin = time.time()
EmbCacheLib.cudaSplit(index, _total_mask)
end = time.time()
print(end - begin)

EmbCacheLib.cudaUnpin(_total_mask)
begin = time.time()
EmbCacheLib.cudaSplit(index, _total_mask)
end = time.time()
print(end - begin)