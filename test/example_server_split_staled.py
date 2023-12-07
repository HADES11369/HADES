import torch
import time

len = 400000
total = 4000000
dim = 128
idx = torch.randint(0, total, (len, )).long().unique()
len = idx.shape[0]
grad = torch.randn((len, dim)).float()
cnt = torch.randint(0, 16, (len, )).long()

num_trainers = 8

for i in range(10):
    tic = time.time()

    split_idx_list = []
    split_grad_list = []
    split_cnt_list = []

    split = idx % num_trainers
    for rank in range(num_trainers):
        local_mask = split == rank
        local_index = local_mask.nonzero().flatten()
        local_size = local_index.shape[0]
        part_remote_idx = torch.zeros((local_size, ), dtype=idx.dtype)
        part_remote_grad = torch.zeros((local_size, dim), dtype=grad.dtype)
        part_remote_cnt = torch.zeros((local_size, ), dtype=cnt.dtype)
        part_remote_idx[:] = idx[local_index]
        part_remote_grad[:] = grad[local_index]
        part_remote_cnt[:] = cnt[local_index]
        split_idx_list.append(part_remote_idx)
        split_grad_list.append(part_remote_grad)
        split_cnt_list.append(part_remote_cnt)

    toc = time.time()

    print((toc - tic) * 1000)
