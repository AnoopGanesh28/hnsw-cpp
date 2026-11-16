// Ground-truth tool.
//
//   bench_gt check  --base F --queries F --gt F --n N --nq N --dim D --k K [--json OUT]
//     Reads the three files with the library's readers (hnsw::read_fvecs /
//     read_ivecs) and checks their shapes and that every ground-truth id is in
//     [0, n) and unique within its row.
//
//   bench_gt verify --base F --queries F --gt F [--sample 100] [--threads 0] [--json OUT]
//     Recomputes the exact top-K (K = ground-truth row length) for `sample`
//     queries spread evenly over the query file, by multi-threaded brute force,
//     and compares DISTANCES rank by rank with the distances of the provided
//     ids (recomputed with the same kernel), so ties may permute ids without
//     counting as errors. Exits 1 if any rank differs.
//
//   bench_gt subset --base F --queries F --n N --out GT.ivecs [--k 100] [--threads 0] [--json OUT]
//     Exact top-k of every query among the first n base vectors, written as
//     ivecs, for dev-scale runs on a prefix of the base set.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "args.h"
#include "dataset.h"
#include "env.h"
#include "exact.h"
#include "hnsw/distance.h"
#include "hnsw/io.h"
#include "hnsw/thread_pool.h"
#include "json.h"
#include "tool.h"

namespace {

using bench::json::Value;

const char* kUsage =
    "usage: bench_gt check|verify|subset [--flag value ...] (see bench/README.md)\n";

Value dataset_block(const std::vector<std::string>& files) {
  const bench::Manifest m = bench::manifest_for(files.front());
  Value d = Value::object();
  d.set("name", m.dataset);
  Value list = Value::array();
  for (const std::string& f : files) list.push(bench::file_record(f, m));
  d.set("files", list);
  return d;
}

void maybe_write(bench::Args& args, Value doc) {
  if (!args.has("json")) return;
  bench::json::write_file(args.str("json"), doc);
}

int check(bench::Args& args) {
  const std::string base_path = args.str("base");
  const std::string query_path = args.str("queries");
  const std::string gt_path = args.str("gt");
  const size_t n = args.size("n"), nq = args.size("nq"), dim = args.size("dim"),
               k = args.size("k");
  args.has("json");
  args.finish();

  const hnsw::FvecsData base = hnsw::read_fvecs(base_path);
  const hnsw::FvecsData queries = hnsw::read_fvecs(query_path);
  const hnsw::IvecsData gt = hnsw::read_ivecs(gt_path);

  std::vector<std::string> problems;
  auto expect_shape = [&](const char* what, size_t count, size_t d, size_t want_count,
                          size_t want_d) {
    if (count != want_count || d != want_d) {
      problems.push_back(std::string(what) + " is " + std::to_string(count) + " x " +
                         std::to_string(d) + ", expected " + std::to_string(want_count) +
                         " x " + std::to_string(want_d));
    }
  };
  expect_shape("base", base.count, base.dim, n, dim);
  expect_shape("queries", queries.count, queries.dim, nq, dim);
  expect_shape("ground truth", gt.count, gt.dim, nq, k);

  size_t out_of_range = 0, duplicate_rows = 0;
  for (size_t q = 0; q < gt.count; ++q) {
    std::set<int32_t> seen;
    for (size_t j = 0; j < gt.dim; ++j) {
      const int32_t id = gt.data[q * gt.dim + j];
      if (id < 0 || static_cast<size_t>(id) >= base.count) ++out_of_range;
      seen.insert(id);
    }
    if (seen.size() != gt.dim) ++duplicate_rows;
  }
  if (out_of_range) {
    problems.push_back(std::to_string(out_of_range) + " ground-truth ids outside [0, " +
                       std::to_string(base.count) + ")");
  }
  if (duplicate_rows) {
    problems.push_back(std::to_string(duplicate_rows) + " ground-truth rows repeat an id");
  }

  std::printf("base %zu x %zu, queries %zu x %zu, ground truth %zu x %zu\n", base.count,
              base.dim, queries.count, queries.dim, gt.count, gt.dim);
  for (const std::string& p : problems) std::printf("FAIL: %s\n", p.c_str());
  if (problems.empty()) {
    std::printf("OK: shapes as expected, all ground-truth ids in [0, %zu), no repeats\n",
                base.count);
  }

  Value result = Value::object();
  result.set("base_shape", Value::array_of(std::vector<size_t>{base.count, base.dim}));
  result.set("query_shape", Value::array_of(std::vector<size_t>{queries.count, queries.dim}));
  result.set("gt_shape", Value::array_of(std::vector<size_t>{gt.count, gt.dim}));
  result.set("gt_ids_out_of_range", out_of_range);
  result.set("gt_rows_with_repeated_ids", duplicate_rows);
  result.set("ok", problems.empty());
  Value doc = Value::object();
  doc.set("tool", "bench_gt check");
  doc.set("environment", bench::environment_record(dim));
  doc.set("dataset", dataset_block({base_path, query_path, gt_path}));
  doc.set("result", result);
  maybe_write(args, doc);
  return problems.empty() ? 0 : 1;
}

int verify(bench::Args& args) {
  const std::string base_path = args.str("base");
  const std::string query_path = args.str("queries");
  const std::string gt_path = args.str("gt");
  const size_t sample = args.size("sample", 100);
  const size_t threads = args.size("threads", 0);
  args.has("json");
  args.finish();

  const hnsw::FvecsData base = hnsw::read_fvecs(base_path);
  const hnsw::FvecsData queries = hnsw::read_fvecs(query_path);
  const hnsw::IvecsData gt = hnsw::read_ivecs(gt_path);
  if (queries.dim != base.dim) throw std::runtime_error("query and base dimensions differ");
  if (gt.count != queries.count) throw std::runtime_error("ground truth and query counts differ");
  if (sample == 0 || sample > queries.count) {
    throw std::invalid_argument("--sample must be in [1, number of queries]");
  }
  const size_t k = gt.dim;
  const size_t dim = base.dim;

  std::vector<size_t> ids(sample);
  for (size_t i = 0; i < sample; ++i) ids[i] = i * queries.count / sample;

  hnsw::ThreadPool pool(threads);
  const auto t0 = bench::Clock::now();
  const std::vector<hnsw::KnnResult> exact =
      bench::exact_knn(base.data.data(), base.count, dim, queries.data.data(), ids, k, pool);
  const double brute_seconds = bench::seconds_between(t0, bench::Clock::now());

  const hnsw::DistFn dist = hnsw::best_l2_squared();
  size_t bad_queries = 0, bad_ranks = 0, permuted_ranks = 0, bad_ids = 0;
  double max_abs_diff = 0.0;
  Value examples = Value::array();
  for (size_t i = 0; i < sample; ++i) {
    const size_t q = ids[i];
    const float* query = queries.data.data() + q * dim;
    bool query_bad = false;
    for (size_t j = 0; j < k; ++j) {
      const int32_t id = gt.data[q * k + j];
      if (id < 0 || static_cast<size_t>(id) >= base.count) {
        ++bad_ids;
        query_bad = true;
        continue;
      }
      const float d_gt = dist(query, base.data.data() + static_cast<size_t>(id) * dim, dim);
      const float d_exact = exact[i].dists[j];
      if (d_gt != d_exact) {
        ++bad_ranks;
        query_bad = true;
        max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::abs(d_gt - d_exact)));
        if (examples.size() < 10) {
          Value e = Value::object();
          e.set("query", q);
          e.set("rank", j);
          e.set("gt_id", id);
          e.set("gt_distance", static_cast<double>(d_gt));
          e.set("exact_id", exact[i].ids[j]);
          e.set("exact_distance", static_cast<double>(d_exact));
          examples.push(e);
        }
      } else if (static_cast<uint32_t>(id) != exact[i].ids[j]) {
        ++permuted_ranks;
      }
    }
    if (query_bad) ++bad_queries;
  }

  std::printf(
      "verified %zu queries x top-%zu against exact brute force (%.3f s on %zu threads)\n"
      "distance mismatches: %zu ranks in %zu queries (max |diff| %.9g); ids out of range: %zu\n"
      "ranks with equal distance but a different id (tie order): %zu\n",
      sample, k, brute_seconds, pool.size(), bad_ranks, bad_queries, max_abs_diff, bad_ids,
      permuted_ranks);
  const bool ok = bad_ranks == 0 && bad_ids == 0;
  std::printf("%s\n", ok ? "OK: provided ground truth matches exact search"
                         : "FAIL: provided ground truth differs from exact search");

  Value config = Value::object();
  config.set("sampled_queries", Value::array_of(ids));
  config.set("k", k);
  config.set("threads", pool.size());
  config.set("comparison",
             "rank-by-rank distance equality; ids may differ only where distances tie");
  Value result = Value::object();
  result.set("mismatched_queries", bad_queries);
  result.set("mismatched_ranks", bad_ranks);
  result.set("gt_ids_out_of_range", bad_ids);
  result.set("max_abs_distance_diff", max_abs_diff);
  result.set("tie_permuted_ranks", permuted_ranks);
  result.set("examples", examples);
  result.set("brute_force_seconds", brute_seconds);
  result.set("ok", ok);
  Value doc = Value::object();
  doc.set("tool", "bench_gt verify");
  doc.set("environment", bench::environment_record(dim));
  doc.set("dataset", dataset_block({base_path, query_path, gt_path}));
  doc.set("config", config);
  doc.set("result", result);
  maybe_write(args, doc);
  return ok ? 0 : 1;
}

int subset(bench::Args& args) {
  const std::string base_path = args.str("base");
  const std::string query_path = args.str("queries");
  const size_t n = args.size("n");
  const std::string out = args.str("out");
  const size_t k = args.size("k", 100);
  const size_t threads = args.size("threads", 0);
  args.has("json");
  args.finish();
  if (k == 0 || k > n) throw std::invalid_argument("--k must be in [1, n]");

  const hnsw::FvecsData base = bench::read_fvecs_prefix(base_path, n);
  const hnsw::FvecsData queries = hnsw::read_fvecs(query_path);
  if (queries.dim != base.dim) throw std::runtime_error("query and base dimensions differ");

  std::vector<size_t> ids(queries.count);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = i;
  hnsw::ThreadPool pool(threads);
  const auto t0 = bench::Clock::now();
  const std::vector<hnsw::KnnResult> exact = bench::exact_knn(
      base.data.data(), base.count, base.dim, queries.data.data(), ids, k, pool);
  const double seconds = bench::seconds_between(t0, bench::Clock::now());

  std::vector<int32_t> flat;
  flat.reserve(queries.count * k);
  for (const hnsw::KnnResult& r : exact) {
    for (uint32_t id : r.ids) flat.push_back(static_cast<int32_t>(id));
  }
  hnsw::write_ivecs(out, flat, k);
  std::printf("wrote %s: exact top-%zu of %zu queries among the first %zu base vectors "
              "(%.3f s on %zu threads)\n",
              out.c_str(), k, queries.count, n, seconds, pool.size());

  Value config = Value::object();
  config.set("base_prefix", n);
  config.set("k", k);
  config.set("threads", pool.size());
  Value doc = Value::object();
  doc.set("tool", "bench_gt subset");
  doc.set("environment", bench::environment_record(base.dim));
  doc.set("dataset", dataset_block({base_path, query_path}));
  doc.set("config", config);
  doc.set("output", bench::file_record(out, bench::manifest_for(base_path)));
  doc.set("brute_force_seconds", seconds);
  maybe_write(args, doc);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  return bench::run_tool("bench_gt", [&] {
    if (argc < 2) {
      std::fputs(kUsage, stderr);
      return 2;
    }
    const std::string mode = argv[1];
    bench::Args args(argc, argv, 2);
    if (mode == "check") return check(args);
    if (mode == "verify") return verify(args);
    if (mode == "subset") return subset(args);
    std::fputs(kUsage, stderr);
    return 2;
  });
}
