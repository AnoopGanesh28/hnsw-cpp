// Generates a seeded synthetic dataset in the SIFT1M layout, for use ONLY when
// the real SIFT1M download is impossible (and for small test datasets).
//
//   bench_make_synthetic --out DIR [--n 1000000] [--nq 10000] [--dim 128] [--k 100]
//                        [--clusters 100] [--center-std 3] [--point-std 1]
//                        [--seed 1] [--threads 0]
//
// Data: a Gaussian mixture. `clusters` centers are drawn from
// N(0, center_std^2 I); every base vector and every query picks a center
// uniformly and adds N(0, point_std^2 I) noise. Queries are drawn from the same
// mixture after the base set. Ground truth is the exact top-k of every query,
// by multi-threaded brute force. Output in DIR: synthetic_base.fvecs,
// synthetic_query.fvecs, synthetic_groundtruth.ivecs, and a MANIFEST whose
// dataset name is "synthetic", which every benchmark result file then carries.
//
// The random streams are std::mt19937_64 with std::normal_distribution and
// std::uniform_int_distribution, so the bytes are reproducible for a given seed
// with the same standard library (the MANIFEST records which one).
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "args.h"
#include "dataset.h"
#include "exact.h"
#include "hnsw/io.h"
#include "hnsw/thread_pool.h"
#include "tool.h"

namespace {

std::string stdlib_name() {
#if defined(_GLIBCXX_RELEASE)
  return "libstdc++ " + std::to_string(_GLIBCXX_RELEASE);
#elif defined(_LIBCPP_VERSION)
  return "libc++ " + std::to_string(_LIBCPP_VERSION);
#else
  return "unknown";
#endif
}

std::string fmt(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

int run(int argc, char** argv) {
  bench::Args args(argc, argv);
  const std::string out = args.str("out");
  const size_t n = args.size("n", 1000000);
  const size_t nq = args.size("nq", 10000);
  const size_t dim = args.size("dim", 128);
  const size_t k = args.size("k", 100);
  const size_t clusters = args.size("clusters", 100);
  const double center_std = args.real("center-std", 3.0);
  const double point_std = args.real("point-std", 1.0);
  const size_t seed = args.size("seed", 1);
  const size_t threads = args.size("threads", 0);
  args.finish();
  if (n == 0 || nq == 0 || dim == 0 || clusters == 0) {
    throw std::invalid_argument("--n, --nq, --dim and --clusters must be positive");
  }
  if (k == 0 || k > n) throw std::invalid_argument("--k must be in [1, n]");
  if (!(center_std > 0) || !(point_std > 0)) {
    throw std::invalid_argument("--center-std and --point-std must be positive");
  }

  std::mt19937_64 rng(seed);
  std::normal_distribution<float> center_noise(0.0f, static_cast<float>(center_std));
  std::normal_distribution<float> point_noise(0.0f, static_cast<float>(point_std));
  std::uniform_int_distribution<size_t> pick(0, clusters - 1);

  std::vector<float> centers(clusters * dim);
  for (float& x : centers) x = center_noise(rng);
  auto draw = [&](size_t count) {
    std::vector<float> v(count * dim);
    for (size_t i = 0; i < count; ++i) {
      const float* c = centers.data() + pick(rng) * dim;
      for (size_t d = 0; d < dim; ++d) v[i * dim + d] = c[d] + point_noise(rng);
    }
    return v;
  };
  const std::vector<float> base = draw(n);
  const std::vector<float> queries = draw(nq);

  std::vector<size_t> ids(nq);
  for (size_t i = 0; i < nq; ++i) ids[i] = i;
  hnsw::ThreadPool pool(threads);
  const std::vector<hnsw::KnnResult> exact =
      bench::exact_knn(base.data(), n, dim, queries.data(), ids, k, pool);
  std::vector<int32_t> gt;
  gt.reserve(nq * k);
  for (const hnsw::KnnResult& r : exact) {
    for (uint32_t id : r.ids) gt.push_back(static_cast<int32_t>(id));
  }

  std::filesystem::create_directories(out);
  const std::vector<std::string> files = {"synthetic_base.fvecs", "synthetic_query.fvecs",
                                          "synthetic_groundtruth.ivecs"};
  const std::filesystem::path dir(out);
  hnsw::write_fvecs((dir / files[0]).string(), base, dim);
  hnsw::write_fvecs((dir / files[1]).string(), queries, dim);
  hnsw::write_ivecs((dir / files[2]).string(), gt, k);
  bench::write_manifest(out, "synthetic",
                        {{"generator", "bench_make_synthetic (Gaussian mixture)"},
                         {"n", std::to_string(n)},
                         {"nq", std::to_string(nq)},
                         {"dim", std::to_string(dim)},
                         {"k", std::to_string(k)},
                         {"clusters", std::to_string(clusters)},
                         {"center_std", fmt(center_std)},
                         {"point_std", fmt(point_std)},
                         {"seed", std::to_string(seed)},
                         {"random", "std::mt19937_64, " + stdlib_name()},
                         {"ground_truth", "exact brute force (hnsw::brute_force_knn)"}},
                        files);
  std::printf("wrote synthetic dataset to %s: base %zu x %zu, queries %zu, top-%zu ground truth\n",
              out.c_str(), n, dim, nq, k);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  return bench::run_tool("bench_make_synthetic", [&] { return run(argc, argv); });
}
