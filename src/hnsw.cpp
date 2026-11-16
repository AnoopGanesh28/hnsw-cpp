#include "hnsw/hnsw.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "search_layer.h"

namespace hnsw {

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
  layout_ = Layout::make(dim, params.M);
  ml_ = 1.0 / std::log(static_cast<double>(params.M));
}

void HnswIndex::reserve(size_t n) {
  arena_.reserve(n * layout_.record_stride);
  levels_.reserve(n);
  upper_offsets_.reserve(n);
}

GraphView HnswIndex::view() const {
  GraphView g;
  g.layout = layout_;
  g.base = arena_.data();
  g.count = size();
  g.levels = levels_.data();
  g.upper_offsets = upper_offsets_.data();
  g.upper_links = upper_links_.data();
  g.entry_point = entry_;
  g.max_level = max_level_;
  return g;
}

const float* HnswIndex::vec(uint32_t id) const {
  return reinterpret_cast<const float*>(arena_.data() + layout_.record_offset(id) +
                                        layout_.vector_offset);
}

uint32_t* HnswIndex::link_block(uint32_t id, int level) {
  if (level == 0) {
    return reinterpret_cast<uint32_t*>(arena_.data() + layout_.record_offset(id) +
                                       layout_.count_offset);
  }
  return upper_links_.data() + layout_.upper_block_offset(upper_offsets_[id], level);
}

// level = floor(-ln(U) * mL), U uniform in (0, 1]. U is built from the top
// 53 bits of the generator so the sequence is identical on every platform.
int HnswIndex::sample_level() {
  const double u = static_cast<double>(rng_() >> 11) * 0x1.0p-53;  // [0, 1)
  return static_cast<int>(std::floor(-std::log(1.0 - u) * ml_));
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

// Overwrites the neighbor list of id on `level`; unused slots are zeroed so
// the stored bytes depend only on the list contents.
void HnswIndex::set_links(uint32_t id, int level, const std::vector<uint32_t>& ids) {
  uint32_t* block = link_block(id, level);
  const size_t cap = layout_.max_links(level);
  block[0] = static_cast<uint32_t>(ids.size());
  std::copy(ids.begin(), ids.end(), block + 1);
  std::fill(block + 1 + ids.size(), block + 1 + cap, 0u);
}

// Adds the link from -> to on `level`; if from's list then exceeds its cap,
// re-selects it with the heuristic over (its neighbors + to).
void HnswIndex::connect(uint32_t from, uint32_t to, int level) {
  uint32_t* block = link_block(from, level);
  const size_t count = block[0];
  const size_t cap = layout_.max_links(level);
  if (count < cap) {
    block[1 + count] = to;
    block[0] = static_cast<uint32_t>(count + 1);
    return;
  }
  std::vector<Cand> cands;
  cands.reserve(count + 1);
  for (size_t i = 0; i < count; ++i) cands.push_back({distance(vec(from), block[1 + i]), block[1 + i]});
  cands.push_back({distance(vec(from), to), to});
  std::sort(cands.begin(), cands.end());
  set_links(from, level, select_neighbors(cands, cap));
}

// Algorithm 1 (INSERT).
uint32_t HnswIndex::add(const float* v) {
  if (size() >= std::numeric_limits<uint32_t>::max()) {
    throw std::length_error("HnswIndex: too many elements for uint32 ids");
  }
  const uint32_t id = static_cast<uint32_t>(size());
  const int level = sample_level();

  arena_.resize(arena_.size() + layout_.record_stride);
  std::memcpy(arena_.data() + layout_.record_offset(id) + layout_.vector_offset, v,
              dim_ * sizeof(float));
  levels_.push_back(static_cast<uint8_t>(level));
  if (level > 0) {
    upper_offsets_.push_back(upper_links_.size());
    upper_links_.resize(upper_links_.size() + static_cast<size_t>(level) * layout_.upper_block, 0u);
  } else {
    upper_offsets_.push_back(0);
  }

  if (id == 0) {
    entry_ = 0;
    max_level_ = level;
    return id;
  }

  const float* q = vec(id);
  const GraphView g = view();
  Cand cur{distance(q, entry_), entry_};
  for (int lc = max_level_; lc > level; --lc) cur = detail::greedy_closest(g, dist_, q, cur, lc);

  std::vector<Cand> eps{cur};
  for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
    const detail::CandidateSpan found = detail::search_layer(
        g, dist_, q, eps.data(), eps.size(), params_.ef_construction, lc, insert_scratch_);
    eps.assign(found.data, found.data + found.size);
    const std::vector<uint32_t> selected = select_neighbors(eps, params_.M);
    set_links(id, lc, selected);
    for (uint32_t e : selected) connect(e, id, lc);
  }

  if (level > max_level_) {
    max_level_ = level;
    entry_ = id;
  }
  return id;
}

KnnResult HnswIndex::search(const float* query, size_t k, size_t ef) const {
  thread_local SearchScratch scratch;
  return search(query, k, ef, scratch);
}

KnnResult HnswIndex::search(const float* query, size_t k, size_t ef,
                            SearchScratch& scratch) const {
  return search_graph(view(), dist_, query, k, ef, scratch);
}

size_t HnswIndex::search_into(const float* query, size_t k, size_t ef, SearchScratch& scratch,
                              uint32_t* out_ids, float* out_dists) const {
  return search_graph_into(view(), dist_, query, k, ef, scratch, out_ids, out_dists);
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
  return view().neighbors(id, level);
}

std::string HnswIndex::check_invariants() const {
  std::ostringstream err;
  const size_t n = size();
  if (arena_.size() != n * layout_.record_stride || upper_offsets_.size() != n) {
    err << "storage sizes disagree with node count " << n;
    return err.str();
  }
  if (reinterpret_cast<uintptr_t>(arena_.data()) % Layout::kRecordAlign != 0) {
    err << "arena is not " << Layout::kRecordAlign << "-byte aligned";
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
    err << "entry point " << entry_ << " has level " << int{levels_[entry_]}
        << ", max_level is " << max_level_;
    return err.str();
  }
  const GraphView g = view();
  for (uint32_t id = 0; id < n; ++id) {
    const int lvl = levels_[id];
    if (lvl > max_level_) {
      err << "node " << id << " has level " << lvl << " outside [0, " << max_level_ << "]";
      return err.str();
    }
    if (lvl > 0 && upper_offsets_[id] + static_cast<uint64_t>(lvl) * layout_.upper_block >
                       upper_links_.size()) {
      err << "node " << id << " upper links exceed the upper table";
      return err.str();
    }
    for (int l = 0; l <= lvl; ++l) {
      const NeighborView list = g.neighbors(id, l);
      if (list.count > layout_.max_links(l)) {
        err << "node " << id << " level " << l << " degree " << list.count
            << " exceeds cap " << layout_.max_links(l);
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
              << " whose level is " << int{levels_[nb]};
          return err.str();
        }
      }
      std::vector<uint32_t> sorted(list.begin(), list.end());
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
