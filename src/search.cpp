#include <algorithm>
#include <functional>
#include <queue>

#include "hnsw/graph.h"
#include "search_layer.h"

namespace hnsw {
namespace detail {
namespace {

using MaxHeap = std::priority_queue<Cand>;
using MinHeap = std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>>;

}  // namespace

Cand greedy_closest(const GraphView& g, DistFn dist, const float* q, Cand cur, int level) {
  const size_t dim = g.layout.dim;
  bool improved = true;
  while (improved) {
    improved = false;
    for (uint32_t e : g.neighbors(cur.second, level)) {
      const Cand c{dist(q, g.vector(e), dim), e};
      if (c < cur) {
        cur = c;
        improved = true;
      }
    }
  }
  return cur;
}

std::vector<Cand> search_layer(const GraphView& g, DistFn dist, const float* q,
                               const std::vector<Cand>& entry, size_t ef, int level) {
  const size_t dim = g.layout.dim;
  std::vector<uint8_t> visited(g.count, 0);
  MinHeap candidates;
  MaxHeap found;
  for (const Cand& e : entry) {
    if (visited[e.second]) continue;
    visited[e.second] = 1;
    candidates.push(e);
    found.push(e);
    if (found.size() > ef) found.pop();
  }
  while (!candidates.empty()) {
    const Cand c = candidates.top();
    if (c.first > found.top().first) break;
    candidates.pop();
    for (uint32_t e : g.neighbors(c.second, level)) {
      if (visited[e]) continue;
      visited[e] = 1;
      const Cand cand{dist(q, g.vector(e), dim), e};
      if (found.size() < ef || cand < found.top()) {
        candidates.push(cand);
        found.push(cand);
        if (found.size() > ef) found.pop();
      }
    }
  }
  std::vector<Cand> out(found.size());
  for (size_t i = out.size(); i-- > 0;) {
    out[i] = found.top();
    found.pop();
  }
  return out;
}

}  // namespace detail

KnnResult search_graph(const GraphView& g, DistFn dist, const float* query,
                       size_t k, size_t ef) {
  using detail::Cand;
  KnnResult result;
  if (g.count == 0 || k == 0) return result;

  Cand cur{dist(query, g.vector(g.entry_point), g.layout.dim), g.entry_point};
  for (int lc = g.max_level; lc > 0; --lc) cur = detail::greedy_closest(g, dist, query, cur, lc);
  const std::vector<Cand> w = detail::search_layer(g, dist, query, {cur}, std::max(ef, k), 0);

  const size_t take = std::min(k, w.size());
  result.ids.reserve(take);
  result.dists.reserve(take);
  for (size_t i = 0; i < take; ++i) {
    result.dists.push_back(w[i].first);
    result.ids.push_back(w[i].second);
  }
  return result;
}

}  // namespace hnsw
