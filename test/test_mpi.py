from mpi4py import MPI
import numpy as np
import torch
import EmbCacheLib

NUM_PROCESS = 3

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert size == NUM_PROCESS

# all2all
all2all_input = torch.arange(NUM_PROCESS).float()
all2all_output = torch.empty_like(all2all_input)

comm.Alltoall(all2all_input, all2all_output)
#print(rank, all2all_input)
#print(rank, all2all_output)

#print()
#exit()
# all2allv
all2allv_input_split = torch.arange(NUM_PROCESS).int().tolist()
all2allv_output_split = [rank for _ in range(NUM_PROCESS)]

all2allv_input = []
for i in range(NUM_PROCESS):
    all2allv_input += [i for _ in range(i)]

all2allv_input = torch.tensor(all2allv_input).float()
all2allv_output = torch.zeros(sum(all2allv_output_split) * 2).float()
comm.Alltoallv([all2allv_input, all2allv_input_split],
               [all2allv_output, all2allv_output_split])
#print(rank, all2allv_input)
print(rank, all2allv_output)
