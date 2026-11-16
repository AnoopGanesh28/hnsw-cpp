#include <algorithm>
#include <limits>

#include "hnsw/graph.h"
#include "hnsw/thread_pool.h"

namespace hnsw {

void search_graph_batch(const GraphView& graph, DistFn dist, const float* queries,
                        size_t n_queries, size_t k, size_t ef, uint32_t* out_ids,
                        float* out_dists, ThreadPool& pool, size_t chunk) {
  const size_t dim = graph.layout.dim;
  pool.parallel_for(n_queries, chunk, [&](size_t thread, size_t begin, size_t end) {
    SearchScratch& scratch = pool.scratch(thread);
    for (size_t q = begin; q < end; ++q) {
      uint32_t* ids = out_ids + q * k;
      float* dists = out_dists + q * k;
      const size_t got =
          search_graph_into(graph, dist, queries + q * dim, k, ef, scratch, ids, dists);
      std::fill(ids + got, ids + k, kNoNeighbor);
      std::fill(dists + got, dists + k, std::numeric_limits<float>::infinity());
    }
  });
}

}  // namespace hnsw
