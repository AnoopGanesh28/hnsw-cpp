// Fuzz target for the index loader. The input is an index file image.
//
// It goes through LoadedIndex::from_memory, the validation and view
// construction that LoadedIndex::open runs on every file, in a 64-byte-aligned
// buffer of exactly the input's size, so ASan flags any read past its end.
// If deep validation accepts it, the graph is walked through the accessors
// and searched (fixed and input-derived queries, several k and ef, a reused
// SearchScratch, search_into, and search_batch on a small pool): deep
// validation promises all of that is memory-safe. Shallow-validated images
// are never searched (they are trusted by contract).
//
// The same bytes then go through LoadedIndex::open in Read mode, from an
// in-memory file, which must reach the same verdict (the same error message,
// or identical search results).
//
// Built as a libFuzzer target (HNSW_BUILD_FUZZ) and, with replay_main.cpp,
// as the replay_loader regression test.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hnsw/index_file.h"
#include "hnsw/thread_pool.h"
#include "index_file_util.h"
#include "memfd.h"

using hnsw::IndexFileError;
using hnsw::KnnResult;
using hnsw::LoadedIndex;
using hnsw::LoadMode;
using hnsw::LoadOptions;

namespace {

[[noreturn]] void mismatch(const std::string& what) {
  std::fprintf(stderr, "fuzz_loader: %s\n", what.c_str());
  std::abort();
}

// The message without its "hnsw index '<name>': " prefix.
std::string reason(const IndexFileError& e) {
  const std::string m = e.what();
  const size_t p = m.find("': ");
  return p == std::string::npos ? m : m.substr(p + 3);
}

struct Verdict {
  bool loaded = false;
  std::string error;
};

// Reads everything reachable through the accessors. Returns a checksum so
// the reads are not optimized away.
uint64_t walk(const LoadedIndex& index) {
  uint64_t sum = index.size() > 0 ? index.entry_point() : 0;
  for (uint32_t id = 0; id < index.size(); ++id) {
    const int top = index.node_level(id);
    for (int level = 0; level <= top; ++level) {
      for (uint32_t x : index.neighbors(id, level)) sum += x + index.node_level(x);
    }
    const float* v = index.vector(id);
    for (size_t d = 0; d < index.dim(); ++d) {
      uint32_t bits;
      std::memcpy(&bits, v + d, sizeof(bits));
      sum += bits;
    }
  }
  return sum;
}

// Queries: zeros, the entry point's vector, small values derived from the
// input, and raw input bits (which may be NaN or infinite; search must stay
// memory-safe for any query).
std::vector<float> make_queries(const LoadedIndex& index, const uint8_t* data, size_t size,
                                size_t* count) {
  const size_t dim = index.dim();
  *count = 4;
  std::vector<float> q(*count * dim, 0.0f);
  if (index.size() > 0) std::memcpy(&q[dim], index.vector(index.entry_point()), dim * sizeof(float));
  for (size_t d = 0; d < dim; ++d) {
    q[2 * dim + d] = (static_cast<int>(data[(d * 7 + 1) % size]) - 128) * 0.25f;
    uint32_t bits = 0;
    for (size_t b = 0; b < 4; ++b) bits |= uint32_t{data[(d * 4 + b + 3) % size]} << (8 * b);
    std::memcpy(&q[3 * dim + d], &bits, sizeof(bits));
  }
  return q;
}

hnsw::ThreadPool& pool() {
  static hnsw::ThreadPool p(2);
  return p;
}

uint64_t search_all(const LoadedIndex& index, const uint8_t* data, size_t size) {
  static hnsw::SearchScratch scratch;  // reused across inputs of different shapes
  size_t nq = 0;
  const std::vector<float> queries = make_queries(index, data, size, &nq);
  const size_t dim = index.dim();
  const size_t in_k = 1 + data[size - 1] % 64, in_ef = data[size - 2];
  const size_t kef[][2] = {{1, 0}, {1, 1}, {10, 16}, {index.size() + 3, 64}, {in_k, in_ef}};

  uint64_t sum = 0;
  for (size_t q = 0; q < nq; ++q) {
    const float* qp = queries.data() + q * dim;
    for (const auto& p : kef) {
      const KnnResult r = index.search(qp, p[0], p[1], scratch);
      for (uint32_t id : r.ids) sum += id;
    }
  }
  const size_t k = in_k;
  std::vector<uint32_t> ids(nq * k);
  std::vector<float> dists(nq * k);
  for (size_t q = 0; q < nq; ++q) {
    sum += index.search_into(queries.data() + q * dim, k, in_ef, scratch, &ids[q * k],
                             &dists[q * k]);
  }
  index.search_batch(queries.data(), nq, k, in_ef, ids.data(), dists.data(), pool(), 1);
  for (uint32_t id : ids) sum += id;
  return sum;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  const test::AlignedBytes bytes(data, size);
  LoadOptions deep;
  Verdict memory;
  uint64_t memory_sum = 0;
  try {
    const LoadedIndex index = LoadedIndex::from_memory(bytes.data(), bytes.size(), deep);
    memory.loaded = true;
    walk(index);
    memory_sum = search_all(index, data, size);
  } catch (const IndexFileError& e) {
    memory.error = reason(e);
  }

  // Shallow validation accepts a superset: whatever deep rejects in the
  // header, shallow rejects with the same message. Not searched.
  LoadOptions shallow;
  shallow.validate_deep = false;
  try {
    LoadedIndex::from_memory(bytes.data(), bytes.size(), shallow);
  } catch (const IndexFileError& e) {
    if (memory.loaded || reason(e) != memory.error) {
      mismatch("shallow rejected (" + reason(e) + ") but deep said: " +
               (memory.loaded ? "loaded" : memory.error));
    }
  }

  const fuzz::MemFile file(data, size);
  LoadOptions read = deep;
  read.mode = LoadMode::Read;
  try {
    const LoadedIndex index = LoadedIndex::open(file.path(), read);
    if (!memory.loaded) mismatch("Read mode loaded what from_memory rejected: " + memory.error);
    walk(index);
    if (search_all(index, data, size) != memory_sum) mismatch("Read mode searched differently");
  } catch (const IndexFileError& e) {
    if (memory.loaded || reason(e) != memory.error) {
      mismatch("Read mode: " + reason(e) + "; from_memory: " +
               (memory.loaded ? "loaded" : memory.error));
    }
  }
  return 0;
}
