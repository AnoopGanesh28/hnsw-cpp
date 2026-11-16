// Builds one index (one system per process) and records build time, peak
// memory and index file size.
//
//   bench_build --system ours|hnswlib --data BASE.fvecs --out INDEX --json OUT.json
//               [--n all] [--M 16] [--efc 200] [--seed 100] [--runs 1]
//               [--kernel best|scalar]
//
// Insertion is single-threaded for both systems (ours has no concurrent
// insert). Each run builds a fresh index from the same in-memory vectors; the
// build time of a run spans index construction through the last insertion
// (hnswlib preallocates in its constructor, ours in reserve(), so both are
// inside). The index of the last run is saved to INDEX (HnswIndex::save /
// hnswlib saveIndex; save time is recorded separately). VmHWM is read after
// the dataset is loaded and at the end; the latter is the peak over the
// whole process, i.e. dataset plus the largest of the runs. --kernel scalar
// builds ours with the scalar reference L2 kernel, whose graph equals the
// scalar M2 baseline's (layout ablation).
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "args.h"
#include "dataset.h"
#include "env.h"
#include "hnsw/hnsw.h"
#include "hnswlib_index.h"
#include "json.h"
#include "proc.h"
#include "stats.h"
#include "tool.h"

namespace {

using bench::json::Value;

int run(int argc, char** argv) {
  bench::Args args(argc, argv);
  const std::string system = args.choice("system", {"ours", "hnswlib"}, "");
  const std::string data = args.str("data");
  const std::string out = args.str("out");
  const std::string json_out = args.str("json");
  const size_t n_flag = args.size("n", bench::kAllRecords);
  const size_t M = args.size("M", 16);
  const size_t efc = args.size("efc", 200);
  const size_t seed = args.size("seed", 100);
  const size_t runs = args.size("runs", 1);
  const std::string kernel = args.choice("kernel", {"best", "scalar"}, "best");
  args.finish();
  if (kernel != "best" && system != "ours") {
    throw std::invalid_argument("--kernel applies to --system ours only");
  }
  const hnsw::DistFn dist =
      kernel == "scalar" ? hnsw::l2_squared_scalar : hnsw::best_l2_squared();
  if (runs == 0) throw std::invalid_argument("--runs must be >= 1");

  const bench::Manifest manifest = bench::manifest_for(data);
  const auto t_load = bench::Clock::now();
  const hnsw::FvecsData base = bench::read_fvecs_prefix(data, n_flag);
  const double load_seconds = bench::seconds_between(t_load, bench::Clock::now());
  const int64_t hwm_after_load = bench::vm_hwm_kib();
  const size_t n = base.count, dim = base.dim;

  std::vector<double> build_seconds;
  double save_seconds = 0.0;
  for (size_t r = 0; r < runs; ++r) {
    const bool last = r + 1 == runs;
    if (system == "ours") {
      hnsw::HnswParams params;
      params.M = M;
      params.ef_construction = efc;
      params.seed = seed;
      const auto t0 = bench::Clock::now();
      hnsw::HnswIndex index(dim, params, dist);
      index.reserve(n);
      for (size_t i = 0; i < n; ++i) index.add(base.data.data() + i * dim);
      build_seconds.push_back(bench::seconds_between(t0, bench::Clock::now()));
      if (last) {
        const auto t1 = bench::Clock::now();
        index.save(out);
        save_seconds = bench::seconds_between(t1, bench::Clock::now());
      }
    } else {
      const auto t0 = bench::Clock::now();
      bench::HnswlibIndex index(dim, n, M, efc, seed);
      for (size_t i = 0; i < n; ++i) index.add(base.data.data() + i * dim, i);
      build_seconds.push_back(bench::seconds_between(t0, bench::Clock::now()));
      if (last) {
        const auto t1 = bench::Clock::now();
        index.save(out);
        save_seconds = bench::seconds_between(t1, bench::Clock::now());
      }
    }
    std::fprintf(stderr, "bench_build: %s run %zu/%zu: %.3f s\n", system.c_str(), r + 1, runs,
                 build_seconds.back());
  }
  const int64_t hwm_end = bench::vm_hwm_kib();

  Value config = Value::object();
  config.set("system", system);
  config.set("n", n);
  config.set("dim", dim);
  config.set("M", M);
  config.set("ef_construction", efc);
  config.set("seed", seed);
  config.set("runs", runs);
  config.set("insertion", "single-threaded");
  config.set("kernel", system != "ours" ? "hnswlib L2Space"
                       : dist == hnsw::l2_squared_scalar ? "l2_squared_scalar"
                                                         : "l2_squared_avx2");

  Value dataset = Value::object();
  dataset.set("name", manifest.dataset);
  dataset.set("base", bench::file_record(data, manifest));
  dataset.set("base_vectors_used", n);
  dataset.set("prefix_of_file", n_flag != bench::kAllRecords);

  Value memory = Value::object();
  memory.set("vm_hwm_after_load_kib", hwm_after_load);
  memory.set("vm_hwm_end_kib", hwm_end);

  Value index = Value::object();
  index.set("path", std::filesystem::absolute(out).string());
  index.set("bytes", std::filesystem::file_size(out));
  index.set("save_seconds", save_seconds);

  Value doc = Value::object();
  doc.set("tool", "bench_build");
  doc.set("environment", bench::environment_record(dim));
  doc.set("dataset", dataset);
  doc.set("config", config);
  doc.set("dataset_load_seconds", load_seconds);
  doc.set("build_seconds", bench::samples_json(build_seconds));
  doc.set("memory", memory);
  doc.set("index", index);
  bench::json::write_file(json_out, doc);

  std::printf("%s: n=%zu M=%zu efc=%zu build median %.3f s over %zu run(s), index %ju bytes, "
              "VmHWM %lld kB\n",
              system.c_str(), n, M, efc, bench::summarize(build_seconds).median, runs,
              static_cast<uintmax_t>(std::filesystem::file_size(out)),
              static_cast<long long>(hwm_end));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  return bench::run_tool("bench_build", [&] { return run(argc, argv); });
}
