#pragma once

namespace distemb {

#define CACHE_LINE_SIZE 8

template <typename IdType, typename CacheIdxType, int Size>
class cache_line {
  char data[sizeof(IdType) * Size];
  int access_table[Size];

 public:
  cache_line() {
    for (int i = 0; i < Size; i++) {
      set_id(i, -1);
      access_table[i] = 0;
    }
  }

  __host__ __device__ inline void set_id(int idx, IdType id) {
    // set index of embedding
    reinterpret_cast<IdType *>(data)[idx] = id;
    access_table[idx] = 0;
  }

  __host__ __device__ inline IdType get_id(int idx) const {
    // return a index of embedding according to given index in cache line
    return reinterpret_cast<const IdType *>(data)[idx];
  }

  __host__ __device__ inline CacheIdxType compute_loc(int offset, int idx) {
    return offset * Size + idx;
  }

  __host__ __device__ inline bool is_init(int idx) const {
    // if a returned index of embedding is not equal to -1 -> is init
    return this->get_id(idx) != -1;
  }

  __host__ __device__ inline int get_valid_entries() const {
    // return number of valid entries in this cache line
    int valid = 0;
    for (int i = 0; i < Size; i++) {
      valid += is_init(i);
    }
    return valid;
  }

  __host__ __device__ inline int find(IdType id) const {
    // find a index of embedding equal to id in this cache line
    // if so, return the index of it
    // if not, return -1
    for (int i = 0; i < Size; i++) {
      if (get_id(i) == id) return i;
    }
    return -1;
  }

  __host__ __device__ inline int find_empty_entry() const {
    // find an entry in this cache line which is not valid (or id == -1)
    for (int i = 0; i < Size; i++) {
      if (!is_init(i)) return i;
    }
    return -1;
  }

  __host__ __device__ inline std::tuple<int, bool> find_or_empty_entry(
      IdType id) {
    int idx = -1;
    for (int i = 0; i < Size; i++) {
      auto value = get_id(i);
      if (value == id)
        return std::make_tuple(i, true);
      else if (value == -1)
        idx = i;
    }
    return std::make_tuple(idx, false);
  }

  __host__ __device__ inline void access_loc(int idx) {
    // increase access number when accessing
    for (int i = 0; i < Size; i++) {
      access_table[i]++;
    }
    access_table[idx] = 0;
  }

  __device__ inline void increase_access_table(int idx) { access_table[idx]++; }

  __device__ inline void reset_access_table(int idx) { access_table[idx] = 0; }

  __host__ __device__ inline int find_lru() {
    int idx = 0;
    int tmp = access_table[idx];
    for (int i = 0; i < Size; i++) {
      if (access_table[i] > tmp && access_table[i] != 0) {
        idx = i;
        tmp = access_table[i];
      }
    }
    // if tmp == 0, all enrties are new data
    return tmp == 0 ? -1 : idx;
  }
};

}  // namespace distemb