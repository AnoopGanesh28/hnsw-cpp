#include "hnsw/hnsw.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace hnsw {
namespace {

using Cand = std::pair<float, uint32_t>;
// Candidates compare by distance, then id, giving a total order so traversal
// and results are deterministic under distance ties.
using MaxHeap = std::priority_queue<Cand>;
using MinHeap = std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>>;

}  // namespace

HnswIndex::HnswIndex(size_t dim, HnswParams params, DistFn dist)
    : dim_(dim), params_(params), dist_(dist), rng_(params.seed) {
  if (dim < 1) {
    throw std::invalid_argument("HnswIndex: dim must be >= 1");
  }
  if (params.M < 2) {
    throw std::invalid_argument("HnswIndex: M must be >= 2 (got " +
                                std::to_string(params.M) + ")");
  }
  if (params.ef_construction < 1) {
    throw std::invalid_argument("HnswIndex: ef_construction must be >= 1");
  }
  if (dist == nullptr) {
    throw std::invalid_argument("HnswIndex: distance function must not be null");
  }
  ml_ = 1.0 / std::log(static_cast<double>(params.M));
}

// level = floor(-ln(U) * mL), U uniform in (0, 1]. U is built from the top
// 53 bits of the generator so the sequence is identical on every platform.
int HnswIndex::sample_level() {
  const double u = static_cast<double>(rng_() >> 11) * 0x1.0p-53;  // [0, 1)
  return static_cast<int>(std::floor(-std::log(1.0 - u) * ml_));
}

// Greedy descent on one layer (SEARCH-LAYER with ef = 1): move to the closest
// neighbor while that improves the distance.
HnswIndex::Cand HnswIndex::greedy_closest(const float* q, Cand cur, int level) const {
  bool improved = true;
  while (improved) {
    improved = false;
    for (uint32_t e : links_[cur.second][level]) {
      const Cand c{distance(q, e), e};
      if (c < cur) {
        cur = c;
        improved = true;
      }
    }
  }
  return cur;
}

// Algorithm 2 (SEARCH-LAYER). Returns up to ef nearest found, ascending.
std::vector<HnswIndex::Cand> HnswIndex::search_layer(const float* q,
                                                     const std::vector<Cand>& entry,
                                                     size_t ef, int level) const {
  std::vector<uint8_t> visited(size(), 0);
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
    for (uint32_t e : links_[c.second][level]) {
      if (visited[e]) continue;
      visited[e] = 1;
      const Cand cand{distance(q, e), e};
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

// Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) over candidates sorted ascending by
// distance to the base element. A candidate is kept only if it is closer to the
// base element than to every neighbor already selected. extendCandidates is not
// implemented (the paper recommends it only for extremely clustered data).
std::vector<uint32_t> HnswIndex::select_neighbors(const std::vector<Cand>& sorted,
                                                  size_t m) const {
  std::vector<uint32_t> selected;
  std::vector<uint32_t> discarded;
  for (const Cand& c : sorted) {
    if (selected.size() >= m) break;
    bool keep = true;
    for (uint32_t s : selected) {
      if (distance(vec(c.second), s) < c.first) {
        keep = false;
        break;
      }
    }
    (keep ? selected : discarded).push_back(c.second);
  }
  if (params_.keep_pruned) {
    for (uint32_t d : discarded) {
      if (selected.size() >= m) break;
      selected.push_back(d);
    }
  }
  return selected;
}

// Adds the link from -> to on `level`; if from's list then exceeds its cap,
// re-selects it with the heuristic over (its neighbors + to).
void HnswIndex::connect(uint32_t from, uint32_t to, int level) {
  std::vector<uint32_t>& list = links_[from][level];
  list.push_back(to);
  if (list.size() <= cap(level)) return;
  std::vector<Cand> cands;
  cands.reserve(list.size());
  for (uint32_t x : list) cands.push_back({distance(vec(from), x), x});
  std::sort(cands.begin(), cands.end());
  list = select_neighbors(cands, cap(level));
}

// Algorithm 1 (INSERT).
uint32_t HnswIndex::add(const float* v) {
  if (size() >= std::numeric_limits<uint32_t>::max()) {
    throw std::length_error("HnswIndex: too many elements for uint32 ids");
  }
  const uint32_t id = static_cast<uint32_t>(size());
  vectors_.insert(vectors_.end(), v, v + dim_);
  const float* q = vec(id);
  const int level = sample_level();
  levels_.push_back(level);
  links_.emplace_back(static_cast<size_t>(level) + 1);

  if (id == 0) {
    entry_ = 0;
    max_level_ = level;
    return id;
  }

  Cand cur{distance(q, entry_), entry_};
  for (int lc = max_level_; lc > level; --lc) cur = greedy_closest(q, cur, lc);

  std::vector<Cand> eps{cur};
  for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
    std::vector<Cand> w = search_layer(q, eps, params_.ef_construction, lc);
    links_[id][lc] = select_neighbors(w, params_.M);
    for (uint32_t e : links_[id][lc]) connect(e, id, lc);
    eps = std::move(w);
  }

  if (level > max_level_) {
    max_level_ = level;
    entry_ = id;
  }
  return id;
}

// Algorithm 5 (K-NN-SEARCH).
KnnResult HnswIndex::search(const float* query, size_t k, size_t ef) const {
  KnnResult result;
  if (size() == 0 || k == 0) return result;

  Cand cur{distance(query, entry_), entry_};
  for (int lc = max_level_; lc > 0; --lc) cur = greedy_closest(query, cur, lc);
  const std::vector<Cand> w = search_layer(query, {cur}, std::max(ef, k), 0);

  const size_t take = std::min(k, w.size());
  result.ids.reserve(take);
  result.dists.reserve(take);
  for (size_t i = 0; i < take; ++i) {
    result.dists.push_back(w[i].first);
    result.ids.push_back(w[i].second);
  }
  return result;
}

uint32_t HnswIndex::entry_point() const {
  if (size() == 0) throw std::logic_error("HnswIndex: index is empty");
  return entry_;
}

int HnswIndex::node_level(uint32_t id) const {
  if (id >= size()) {
    throw std::out_of_range("HnswIndex: id " + std::to_string(id) + " out of range");
  }
  return levels_[id];
}

NeighborView HnswIndex::neighbors(uint32_t id, int level) const {
  if (level < 0 || level > node_level(id)) {
    throw std::out_of_range("HnswIndex: node " + std::to_string(id) +
                            " has no level " + std::to_string(level));
  }
  const std::vector<uint32_t>& list = links_[id][level];
  return {list.data(), list.size()};
}

std::string HnswIndex::check_invariants() const {
  std::ostringstream err;
  const size_t n = size();
  if (vectors_.size() != n * dim_ || links_.size() != n) {
    err << "storage sizes disagree with node count " << n;
    return err.str();
  }
  if (n == 0) {
    if (max_level_ != -1) err << "empty index has max_level " << max_level_;
    return err.str();
  }
  if (entry_ >= n) {
    err << "entry point " << entry_ << " out of range";
    return err.str();
  }
  if (levels_[entry_] != max_level_) {
    err << "entry point " << entry_ << " has level " << levels_[entry_]
        << ", max_level is " << max_level_;
    return err.str();
  }
  for (uint32_t id = 0; id < n; ++id) {
    const int lvl = levels_[id];
    if (lvl < 0 || lvl > max_level_) {
      err << "node " << id << " has level " << lvl << " outside [0, " << max_level_ << "]";
      return err.str();
    }
    if (links_[id].size() != static_cast<size_t>(lvl) + 1) {
      err << "node " << id << " at level " << lvl << " has " << links_[id].size()
          << " neighbor lists";
      return err.str();
    }
    for (int l = 0; l <= lvl; ++l) {
      const std::vector<uint32_t>& list = links_[id][l];
      if (list.size() > cap(l)) {
        err << "node " << id << " level " << l << " degree " << list.size()
            << " exceeds cap " << cap(l);
        return err.str();
      }
      for (uint32_t nb : list) {
        if (nb >= n) {
          err << "node " << id << " level " << l << " links invalid id " << nb;
          return err.str();
        }
        if (nb == id) {
          err << "node " << id << " level " << l << " has a self-loop";
          return err.str();
        }
        if (levels_[nb] < l) {
          err << "node " << id << " level " << l << " links node " << nb
              << " whose level is " << levels_[nb];
          return err.str();
        }
      }
      std::vector<uint32_t> sorted = list;
      std::sort(sorted.begin(), sorted.end());
      if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        err << "node " << id << " level " << l << " has duplicate neighbors";
        return err.str();
      }
    }
  }
  return err.str();
}

}  // namespace hnsw
