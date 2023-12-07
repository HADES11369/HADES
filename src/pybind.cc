#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "cache/cache.h"
#include "cuda/ops.h"
#include "ipc/cuda_ipc.h"
#include "shm/shared_memory.h"

namespace py = pybind11;
using namespace distemb;

typedef int64_t SACacheIdxType;
typedef int64_t SACacheIdType;

PYBIND11_MODULE(EmbCacheLib, m) {
  py::class_<SACacheIndex<SACacheIdType, SACacheIdxType>>(m, "IndexCache")
      .def(py::init<int64_t, std::string, bool, bool>(), py::arg("cache_size"),
           py::arg("name"), py::arg("create"), py::arg("pin_memory") = true)
      .def("get_size",
           &SACacheIndex<SACacheIdType, SACacheIdxType>::get_cache_size_)
      .def("get_valid_entries_num",
           &SACacheIndex<SACacheIdType, SACacheIdxType>::get_valid_entries_)
      .def("get_capacity",
           &SACacheIndex<SACacheIdType, SACacheIdxType>::get_capacity_)
      .def("get_space",
           &SACacheIndex<SACacheIdType, SACacheIdxType>::get_space_)
      .def("add", &SACacheIndex<SACacheIdType, SACacheIdxType>::add_)
      .def("lookup", &SACacheIndex<SACacheIdType, SACacheIdxType>::lookup_)
      .def("reset", &SACacheIndex<SACacheIdType, SACacheIdxType>::reset_)
      .def("getall", &SACacheIndex<SACacheIdType, SACacheIdxType>::getall_)
      .def("cudalookup",
           &SACacheIndex<SACacheIdType, SACacheIdxType>::cudalookup_);
  //.def("pin", &SACacheIndex<SACacheIdType, SACacheIdxType>::pin_)
  //.def("unpin", &SACacheIndex<SACacheIdType, SACacheIdxType>::unpin_)

  m.def("create_shared_mem", &create_shared_mem, py::arg("name"),
        py::arg("size"), py::arg("pin_memory") = true)
      .def("open_shared_mem", &open_shared_mem, py::arg("name"),
           py::arg("size"), py::arg("pin_memory") = true)
      .def("release_shared_mem", &release_shared_mem, py::arg("name"),
           py::arg("size"), py::arg("ptr"), py::arg("fd"),
           py::arg("pin_memory") = true)
      .def("open_shared_tensor", &open_shared_tensor, py::arg("ptr"),
           py::arg("dtype"), py::arg("shape"));
  m.def("cudaSplit", &cuda::splitCUDA, py::arg("ids"), py::arg("mask"))
      .def("cudaPin", &cuda::PinTensor, py::arg("tensor"))
      .def("cudaUnpin", &cuda::UnpinTensor, py::arg("tensor"))
      .def("cudaFetch", &cuda::CUDAFetch, py::arg("src"), py::arg("src_index"),
           py::arg("dst"), py::arg("dst_index"))
      .def("cudaFetch2", &cuda::CUDAFetch2, py::arg("src"),
           py::arg("src_index"), py::arg("dst"))
      .def("cudaFetchWithCache2", &cuda::CUDAFetchWithCache2,
           py::arg("cpu_src"), py::arg("gpu_src"), py::arg("src_index"),
           py::arg("index_map"), py::arg("dst"))
      .def("cudaSet", &cuda::CUDASet, py::arg("src"), py::arg("dst"),
           py::arg("dst_index"))
      .def("cudaSetWithCache", &cuda::CUDASetWithCache, py::arg("src"),
           py::arg("cpu_dst"), py::arg("gpu_dst"), py::arg("dst_index"),
           py::arg("index_map"))
      .def("cudaCombine", &cuda::CUDACombine, py::arg("src"), py::arg("dst"),
           py::arg("dst_index"))
      .def("cudaPinPtr", &cuda::RegisterPtr, py::arg("ptr"), py::arg("size"))
      .def("cudaUnpinPtr", &cuda::UnregisterPtr, py::arg("ptr"))
      .def("cudaReminder", &cuda::Reminder, py::arg("in"),
           py::arg("local_num_gpus"))
      .def("cudaBinarySearch", &cuda::CUDABinarySearch, py::arg("in"),
           py::arg("range_partition"), py::arg("local_num_gpus"))
      .def("cudaBinarySearch2", &cuda::CUDABinarySearch2, py::arg("in"),
           py::arg("range"));
}
