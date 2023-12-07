import torch
import EmbCacheLib

NUM_PART = 2
LOCAL_NUM_GPU = 4

index = torch.randint(0, 300_0000, (10, )).cuda()
emb = torch.randn(100_0000, 128).cuda()
range_partition = [i * 300_0000 // NUM_PART for i in range(NUM_PART + 1)]
range_partition = torch.tensor(range_partition).cuda()


split = EmbCacheLib.cudaBinarySearch(index, range_partition, LOCAL_NUM_GPU)
print(index)
print(split)
print()

value, indices = torch.sort(split)
partition_range = torch.arange(NUM_PART * LOCAL_NUM_GPU).long().cuda()
print(value)
print(partition_range)
print()

grads_split = EmbCacheLib.cudaBinarySearch2(partition_range, value)
print(grads_split)
grads_split[1:] = grads_split[1:] - grads_split[:-1]
print(grads_split)