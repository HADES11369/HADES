#pragma once

namespace distemb {

void _cudalookup(torch::Tensor ids, torch::Tensor locs, size_t index_size,
                 void* index);

}  // namespace distemb
