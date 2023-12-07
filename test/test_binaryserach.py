import torch
import EmbCacheLib

NUM_PART = 2
LOCAL_NUM_GPU = 4

index = torch.randint(0, 300_0000, (10, )).cuda()
emb = torch.randn(100_0000, 128).cuda()
range_partition = [i * 300_0000 // NUM_PART for i in range(NUM_PART + 1)]
range_partition = torch.tensor(range_partition).cuda()

print(range_partition)

import time

for _ in range(1):
    begin = time.time()
    ids = []
    embs = []
    out = EmbCacheLib.cudaBinarySearch(index, range_partition, LOCAL_NUM_GPU)
    print(index)
    print(out)
    end = time.time()
    # print((end - begin) * 1000)

for _ in range(1):
    begin = time.time()
    ids = []
    embs = []
    out = EmbCacheLib.cudaReminder(index, LOCAL_NUM_GPU)
    print(index)
    print(out)
    end = time.time()
    print((end - begin) * 1000)
