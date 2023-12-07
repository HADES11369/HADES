#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>
#include <torch/script.h>
#include "../utils.h"
#include "cache_line.h"
#include "cuda_runtime.h"

namespace distemb {

namespace cg = cooperative_groups;

template <typename IdType, typename CacheIdxType, int64_t LINE_SIZE>
__global__ void lookupkernel2(
    const IdType* ids, int64_t len, CacheIdxType* locs, size_t index_size,
    cache_line<IdType, CacheIdxType, LINE_SIZE>* index) {

  assert(LINE_SIZE <= 32);

  const int64_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int64_t num_threads = gridDim.x * blockDim.x;
  const int64_t warp_id = thread_id / 32;
  const int64_t lane_id = thread_id % 32;
  const int64_t num_warp = num_threads / 32;

  auto group = cg::tiled_partition<32>(cg::this_thread_block());

  if (lane_id >= LINE_SIZE) return;

  for(int i = warp_id; i < len; i+=num_warp){
    IdType id = ids[i];
    int offset = id % index_size;
    cache_line<IdType, CacheIdxType, LINE_SIZE> line = index[offset];

    int entry_idx = line.get_id(lane_id) == id ? lane_id : -1;
    int max_entry_idx = cg::reduce(group, entry_idx, cg::greater<int>());

    if (max_entry_idx != -1) {
      line.increase_access_table(lane_id);
      line.reset_access_table(max_entry_idx);
      locs[i] = line.compute_loc(offset, max_entry_idx); 
    }
  }
}

template <typename IdType, typename CacheIdxType>
__global__ void lookupkernel(
    const IdType* ids, int64_t len, CacheIdxType* locs, size_t index_size,
    cache_line<IdType, CacheIdxType, CACHE_LINE_SIZE>* index) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) return;
  IdType id = ids[idx];
  cache_line<IdType, CacheIdxType, CACHE_LINE_SIZE> line =
      index[id % index_size];
  int entry_idx = line.find(id);
  if (entry_idx == -1) {
    locs[idx] = -1;
  } else {
    locs[idx] = line.compute_loc(id % index_size, entry_idx);
    line.access_loc(entry_idx);
  }
}

void _cudalookup(torch::Tensor ids, torch::Tensor locs, size_t index_size,
                 void* index) {
  INTEGER_TYPE_SWITCH(ids.dtype(), IdType, {
    INTEGER_TYPE_SWITCH(locs.dtype(), CacheIdxType, {
      int64_t len = ids.numel();
      int64_t block_size = 256;
      int64_t num_blocks = (len + block_size - 1) / block_size;
      /*
      lookupkernel<IdType, CacheIdxType><<<num_blocks, block_size>>>(
          ids.data_ptr<IdType>(), len, locs.data_ptr<CacheIdxType>(),
          index_size,
          (cache_line<IdType, CacheIdxType, CACHE_LINE_SIZE>*)index);
      */

      lookupkernel2<IdType, CacheIdxType, CACHE_LINE_SIZE>
          <<<num_blocks, block_size>>>(
              ids.data_ptr<IdType>(), len, locs.data_ptr<CacheIdxType>(),
              index_size,
              (cache_line<IdType, CacheIdxType, CACHE_LINE_SIZE>*)index);
    });
  });
}

}  // namespace distemb
