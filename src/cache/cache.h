#pragma once
#include <pthread.h>
#include <torch/script.h>
#include <memory>
#include "../shm/shared_memory.h"
#include "cache_line.h"
#include "cache_ops.h"

namespace distemb {

template <typename IdType, typename CacheIdxType>
class SACacheIndex {
  // cache_size: total entries of cache
  // index_size: number of cache lines
  //
  using CacheLineType = cache_line<IdType, CacheIdxType, CACHE_LINE_SIZE>;

 private:
  CacheLineType *index;
  std::shared_ptr<SharedMemory> shared_mem = nullptr;
  size_t index_size;
  CacheIdxType cache_size;
  // std::vector<pthread_rwlock_t> locks;
  std::vector<omp_lock_t> locks;

  inline cache_line<IdType, CacheIdxType, CACHE_LINE_SIZE> &get_line(
      IdType id) {
    return index[id % this->index_size];
  }

  void _init_locks() {
    for (size_t i = 0; i < this->index_size; i++) {
      // pthread_rwlock_t locks = PTHREAD_RWLOCK_INITIALIZER;
      omp_lock_t writelock;
      omp_init_lock(&writelock);
      this->locks.push_back(std::move(writelock));
    }
  }

  void _init_cache() {
    // init cache to (-1, locs++)
    for (size_t i = 0; i < this->index_size; i++) {
      for (int j = 0; j < CACHE_LINE_SIZE; j++) {
        index[i].set_id(j, -1);
      }
    }
    this->cache_size = this->index_size * CACHE_LINE_SIZE;
  }

  // Assume: _add only will be called by one process (server)
  void _add(const IdType *ids, size_t len, CacheIdxType *locs) {
#pragma omp parallel for
    for (size_t i = 0; i < len; i++) {
      // pthread_rwlock_wrlock(&locks[ids[i] % this->index_size]);
      omp_set_lock(&locks[ids[i] % this->index_size]);

      cache_line<IdType, CacheIdxType, CACHE_LINE_SIZE> &line =
          get_line(ids[i]);
      int idx;
      bool is_exist;
      std::tie(idx, is_exist) = line.find_or_empty_entry(ids[i]);
      if (idx >= 0) {
        CacheIdxType cache_loc =
            line.compute_loc(ids[i] % this->index_size, idx);
        if (!is_exist) line.set_id(idx, ids[i]);
        locs[i] = cache_loc;

      } else if (idx == -1) {
        // using LRU
        int idx = line.find_lru();
        if (idx == -1) {
          locs[i] = -1;

        } else {
          CacheIdxType cache_loc =
              line.compute_loc(ids[i] % this->index_size, idx);
          line.set_id(idx, ids[i]);
          locs[i] = cache_loc;
        }

      } else {
        locs[i] = -1;
      }

      // pthread_rwlock_unlock(&locks[ids[i] % this->index_size]);
      omp_unset_lock(&locks[ids[i] % this->index_size]);
    }
  }

  void _lookup(const IdType *ids, int64_t len, CacheIdxType *locs) {
#pragma omp parallel for
    for (int64_t i = 0; i < len; i++) {
      cache_line<IdType, CacheIdxType, CACHE_LINE_SIZE> &line =
          get_line(ids[i]);
      int entry_idx = line.find(ids[i]);
      if (entry_idx == -1) {
        locs[i] = -1;

      } else {
        locs[i] = line.compute_loc(ids[i] % this->index_size, entry_idx);
        line.access_loc(entry_idx);
      }
    }
  }

  void _reset(const IdType *ids, int64_t len) {
#pragma omp parallel for
    for (int64_t i = 0; i < len; i++) {
      cache_line<IdType, CacheIdxType, CACHE_LINE_SIZE> &line =
          get_line(ids[i]);
      int entry_idx = line.find(ids[i]);
      if (entry_idx == -1)
        ;
      else {
        line.set_id(entry_idx, -1);
      }
    }
  }

  void _getall(IdType *ids, CacheIdxType *locs, size_t len) {
    if ((int64_t)len != this->get_valid_entries_()) {
      printf("len: %ld ve: %ld\n", (int64_t)len, this->get_valid_entries_());
    }
    size_t ids_idx = 0;
    for (size_t i = 0; i < this->index_size; i++) {
      for (size_t j = 0; j < CACHE_LINE_SIZE; j++) {
        if (index[i].get_id(j) != -1) {
          ids[ids_idx] = index[i].get_id(j);
          locs[ids_idx] = index[i].compute_loc(i, j);
          ids_idx++;
          if (ids_idx == len) return;
        }
      }
    }
  }

 public:
  explicit SACacheIndex(size_t cache_size) {
    this->index_size = (cache_size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;
    this->index = new CacheLineType[this->index_size];
    this->_init_cache();
    this->_init_locks();
  }

  SACacheIndex(int64_t cache_size, std::string name, bool create = false,
               bool pin_memory = true) {
    this->shared_mem = std::make_shared<SharedMemory>(name, pin_memory);
    this->index_size = (cache_size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;
    size_t shared_mem_size =
        sizeof(this->cache_size) + sizeof(CacheLineType) * this->index_size;
    char *mem;
    if (create)
      mem = static_cast<char *>(this->shared_mem->CreateNew(shared_mem_size));
    else
      mem = static_cast<char *>(this->shared_mem->Open(shared_mem_size));
    auto shared_cache_size = reinterpret_cast<size_t *>(mem);
    mem += sizeof(this->cache_size);
    this->index = reinterpret_cast<CacheLineType *>(mem);
    if (create) {
      for (size_t i = 0; i < this->index_size; i++)
        new (mem + i) CacheLineType();
      this->_init_cache();
      this->_init_locks();
      *shared_cache_size = this->cache_size;
    } else
      this->cache_size = *shared_cache_size;
  }

  ~SACacheIndex() {
    if (!this->shared_mem) {
      delete[] this->index;
      for (size_t i = 0; i < this->index_size; i++) {
        omp_destroy_lock(&locks[i]);
      }
    }
  }

  inline int64_t get_cache_size_() const { return this->cache_size; }

  int64_t get_valid_entries_() const {
    size_t valid = 0;
    for (size_t i = 0; i < this->index_size; i++)
      valid += index[i].get_valid_entries();
    return valid;
  }

  inline size_t get_capacity_() const {
    return this->index_size * CACHE_LINE_SIZE;
  }

  inline size_t get_space_() const {
    return this->index_size *
           sizeof(cache_line<IdType, CacheIdxType, CACHE_LINE_SIZE>);
  }

  torch::Tensor add_(torch::Tensor ids) {
    torch::Tensor locs =
        torch::full_like(ids, -1,
                         torch::TensorOptions()
                             .dtype(torch::CppTypeToScalarType<CacheIdxType>())
                             .device(torch::kCPU));
    IdType *ids_data = ids.data_ptr<IdType>();
    CacheIdxType *locs_data = locs.data_ptr<CacheIdxType>();
    _add(ids_data, ids.size(0), locs_data);
    return locs;
  }

  inline void reset_(torch::Tensor ids) {
    IdType *ids_data = ids.data_ptr<IdType>();
    _reset(ids_data, ids.size(0));
  }

  torch::Tensor lookup_(torch::Tensor ids) {
    torch::Tensor locs =
        torch::full_like(ids, -1,
                         torch::TensorOptions()
                             .dtype(torch::CppTypeToScalarType<CacheIdxType>())
                             .device(torch::kCPU));
    IdType *ids_data = ids.data_ptr<IdType>();
    CacheIdxType *locs_data = locs.data_ptr<CacheIdxType>();
    _lookup(ids_data, ids.size(0), locs_data);
    return locs;
  }

  torch::Tensor cudalookup_(torch::Tensor ids) {
    torch::Tensor locs =
        torch::full_like(ids, -1,
                         torch::TensorOptions()
                             .dtype(torch::CppTypeToScalarType<CacheIdxType>())
                             .device(torch::kCUDA));

    _cudalookup(ids, locs, this->index_size, this->index);
    return locs;
  }

  void pin_() {
    CUDA_CALL(cudaHostRegister(this->index, this->get_space_(),
                               cudaHostRegisterDefault));
  }

  void unpin_() { CUDA_CALL(cudaHostUnregister(this->index)); }

  std::tuple<torch::Tensor, torch::Tensor> getall_(int64_t len) {
    torch::Tensor ids =
        torch::zeros((len), torch::TensorOptions()
                                .dtype(torch::CppTypeToScalarType<IdType>())
                                .device(torch::kCPU));
    torch::Tensor locs =
        torch::full((len), -1,
                    torch::TensorOptions()
                        .dtype(torch::CppTypeToScalarType<CacheIdxType>())
                        .device(torch::kCPU));
    IdType *ids_data = ids.data_ptr<IdType>();
    CacheIdxType *locs_data = locs.data_ptr<CacheIdxType>();
    _getall(ids_data, locs_data, len);
    return std::make_tuple(ids, locs);
  }
};

}  // namespace distemb