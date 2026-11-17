// Builds an index from random vectors, searches it, saves it, maps the file
// back in, and checks that the loaded index returns the same results.

#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw/index_file.h"
#include "hnsw/thread_pool.h"

int main() {
  constexpr size_t kDim = 64, kN = 5000, kQueries = 100, kK = 10, kEf = 100;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  std::vector<float> base(kN * kDim), queries(kQueries * kDim);
  for (float& x : base) x = uniform(rng);
  for (float& x : queries) x = uniform(rng);

  hnsw::HnswParams params;
  params.M = 16;
  params.ef_construction = 200;
  hnsw::HnswIndex index(kDim, params);  // squared L2, fastest kernel for this CPU
  for (size_t i = 0; i < kN; ++i) index.add(base.data() + i * kDim);

  // Results are ascending by squared distance; ties go to the lower id.
  const hnsw::KnnResult one = index.search(queries.data(), kK, kEf);
  std::printf("nearest to query 0: id %u, squared distance %g\n", one.ids[0], one.dists[0]);

  // Batch search on a thread pool (0 = one thread per hardware thread).
  hnsw::ThreadPool pool(0);
  std::vector<uint32_t> ids(kQueries * kK);
  std::vector<float> dists(kQueries * kK);
  index.search_batch(queries.data(), kQueries, kK, kEf, ids.data(), dists.data(), pool);

  // Save, then map the file read-only and search it in place. Deep validation
  // (the default) makes any file that loads safe to search.
  const std::string path =
      (std::filesystem::temp_directory_path() / "hnsw_cpp_quickstart.hnsw").string();
  index.save(path);
  const hnsw::LoadedIndex loaded = hnsw::LoadedIndex::open(path);
  std::vector<uint32_t> loaded_ids(kQueries * kK);
  std::vector<float> loaded_dists(kQueries * kK);
  loaded.search_batch(queries.data(), kQueries, kK, kEf, loaded_ids.data(), loaded_dists.data(),
                      pool);
  std::filesystem::remove(path);

  const bool same = ids == loaded_ids && dists == loaded_dists;
  std::printf("loaded index (%zu x %zu) gives identical results: %s\n", loaded.size(), loaded.dim(),
              same ? "true" : "false");
  if (!same) throw std::runtime_error("loaded index differs from the in-memory index");
  return 0;
}
