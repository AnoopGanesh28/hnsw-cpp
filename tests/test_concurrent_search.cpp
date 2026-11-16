// Several user threads search one shared, memory-mapped LoadedIndex (and one
// HnswIndex) at the same time, each with its own SearchScratch, and get the
// sequential results. Also run under ThreadSanitizer (HNSW_SANITIZE=thread).

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw/index_file.h"
#include "hnsw/thread_pool.h"
#include "index_file_util.h"
#include "test_helper.h"

using namespace hnsw;

namespace {

constexpr size_t kDim = 24, kN = 3000, kQueries = 150, kK = 10, kEf = 48;

std::vector<KnnResult> sequential(const LoadedIndex& index, const std::vector<float>& queries) {
  std::vector<KnnResult> out;
  SearchScratch scratch;
  for (size_t q = 0; q < kQueries; ++q) {
    out.push_back(index.search(queries.data() + q * kDim, kK, kEf, scratch));
  }
  return out;
}

bool same(const KnnResult& a, const KnnResult& b) {
  return a.ids == b.ids && a.dists.size() == b.dists.size() &&
         std::memcmp(a.dists.data(), b.dists.data(), a.dists.size() * sizeof(float)) == 0;
}

// Runs `body(thread_number)` on `threads` user threads at once.
template <typename Body>
void run_threads(int threads, Body body) {
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(body, t);
  for (std::thread& t : pool) t.join();
}

}  // namespace

static void test_shared_loaded_index() {
  const HnswIndex built = test::build_exact(kN, kDim, 12, 60, 5, best_l2_squared());
  const test::TempFile file("concurrent");
  built.save(file.path());
  const LoadedIndex index = LoadedIndex::open(file.path());  // Mmap
  CHECK(index.mode() == LoadMode::Mmap);
  const std::vector<float> queries = test::exact_data(kQueries, kDim, 6);
  const std::vector<KnnResult> want = sequential(index, queries);

  std::atomic<int> mismatches{0};
  run_threads(8, [&](int t) {
    SearchScratch scratch;  // one per thread
    std::vector<uint32_t> ids(kK);
    std::vector<float> dists(kK);
    for (int round = 0; round < 3; ++round) {
      for (size_t i = 0; i < kQueries; ++i) {
        const size_t q = (i + static_cast<size_t>(t) * 17) % kQueries;  // threads disagree on order
        const float* qp = queries.data() + q * kDim;
        mismatches += !same(index.search(qp, kK, kEf, scratch), want[q]);
        mismatches += !same(index.search(qp, kK, kEf), want[q]);  // thread_local scratch
        const size_t got = index.search_into(qp, kK, kEf, scratch, ids.data(), dists.data());
        mismatches += got != want[q].ids.size() ||
                      !std::equal(ids.begin(), ids.begin() + got, want[q].ids.begin());
        mismatches += !same(built.search(qp, kK, kEf, scratch), want[q]);
      }
    }
  });
  CHECK_EQ(mismatches.load(), 0);

  // User threads calling search_batch at once: on a shared pool (calls are
  // serialized) and on pools of their own.
  ThreadPool shared(3);
  run_threads(4, [&](int t) {
    ThreadPool own(2);
    std::vector<uint32_t> ids(kQueries * kK);
    std::vector<float> dists(kQueries * kK);
    for (ThreadPool* pool : {&shared, &own}) {
      index.search_batch(queries.data(), kQueries, kK, kEf, ids.data(), dists.data(), *pool,
                         1 + static_cast<size_t>(t));
      for (size_t q = 0; q < kQueries; ++q) {
        const KnnResult& w = want[q];
        mismatches += !std::equal(w.ids.begin(), w.ids.end(), ids.begin() + q * kK) ||
                      std::memcmp(w.dists.data(), dists.data() + q * kK,
                                  w.dists.size() * sizeof(float)) != 0;
      }
    }
  });
  CHECK_EQ(mismatches.load(), 0);
}

int main() {
  test_shared_loaded_index();
  return test_finish();
}
