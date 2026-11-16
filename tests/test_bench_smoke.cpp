// End-to-end smoke test of the benchmark pipeline on a tiny synthetic dataset:
// generate data, verify and subset its ground truth, build both systems, run a
// small sweep and the load-time mode, then check the result files against
// values recomputed here independently.
//
// Usage: test_bench_smoke BENCH_BIN_DIR
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

#include "hnsw/brute_force.h"
#include "hnsw/distance.h"
#include "hnsw/io.h"
#include "json.h"
#include "proc.h"
#include "test_helper.h"

namespace fs = std::filesystem;
using bench::json::Value;

namespace {

constexpr size_t kN = 3000;      // generated base vectors
constexpr size_t kUsed = 2500;   // prefix the indexes are built on
constexpr size_t kNq = 200;
constexpr size_t kDim = 32;
constexpr size_t kK = 10;
constexpr size_t kReps = 2;
const std::vector<size_t> kEfs = {10, 40, 400};
const std::vector<size_t> kThreads = {1, 3};
const std::vector<std::string> kSystems = {"ours", "hnswlib"};

class TempDir {
 public:
  TempDir()
      : path_(fs::temp_directory_path() / ("hnsw_bench_smoke_" + std::to_string(::getpid()))) {
    fs::remove_all(path_);
    fs::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  std::string operator/(const std::string& name) const { return (path_ / name).string(); }

 private:
  fs::path path_;
};

std::string bin_dir;

bool run(const std::string& tool, const std::vector<std::string>& args) {
  std::string cmd = bench::shell_quote((fs::path(bin_dir) / tool).string());
  for (const std::string& a : args) cmd += " " + bench::shell_quote(a);
  std::fprintf(stderr, "+ %s\n", cmd.c_str());
  const int status = std::system(cmd.c_str());
  ++test::checks();
  if (status != 0) {
    test::fail(__FILE__, __LINE__, "command failed: " + cmd);
    return false;
  }
  return true;
}

bool has_keys(const Value& obj, const std::vector<std::string>& keys) {
  for (const std::string& k : keys) {
    if (!obj.find(k)) {
      std::fprintf(stderr, "missing key '%s'\n", k.c_str());
      return false;
    }
  }
  return true;
}

void check_environment(const Value& env) {
  CHECK(has_keys(env, {"date_utc", "cpu", "mem_total_kib", "kernel", "wsl", "lscpu_head",
                       "compiler", "build", "source", "ours", "hnswlib"}));
  CHECK(!env.at("cpu").at("model").as_string().empty());
  CHECK(env.at("cpu").at("logical_cpus").as_int() >= 1);
  CHECK(env.at("mem_total_kib").as_int() > 0);
  CHECK(env.at("wsl").at("detected").type() == Value::Type::Bool);
  CHECK(env.at("lscpu_head").size() > 0);
  CHECK(!env.at("compiler").at("version").as_string().empty());
  CHECK(env.at("build").at("cxx_flags").as_string().find("-std=c++17") != std::string::npos);
  CHECK(env.at("source").at("commit").as_string().size() == 40);
  const Value& h = env.at("hnswlib");
  CHECK(has_keys(h, {"header_macros", "runtime_checks", "l2_function", "avx_path_active",
                     "tag_pinned", "commit_pinned", "commit_at_build"}));
  CHECK_EQ(h.at("commit_at_build").as_string(), h.at("commit_pinned").as_string());
  CHECK_EQ(h.at("dim").as_int(), static_cast<int64_t>(kDim));
}

void check_samples(const Value& s, size_t count) {
  CHECK(has_keys(s, {"raw", "median", "min", "max"}));
  CHECK_EQ(s.at("raw").size(), count);
  double lo = 1e300, hi = 0;
  for (const Value& v : s.at("raw").items()) {
    CHECK(v.as_double() > 0);
    lo = std::min(lo, v.as_double());
    hi = std::max(hi, v.as_double());
  }
  CHECK_EQ(s.at("min").as_double(), lo);
  CHECK_EQ(s.at("max").as_double(), hi);
  CHECK(lo <= s.at("median").as_double() && s.at("median").as_double() <= hi);
}

void check_build(const std::string& path, const std::string& system, const std::string& index) {
  const Value doc = bench::json::read_file(path);
  CHECK(has_keys(doc, {"tool", "environment", "dataset", "config", "build_seconds", "memory",
                       "index", "dataset_load_seconds"}));
  check_environment(doc.at("environment"));
  CHECK_EQ(doc.at("dataset").at("name").as_string(), std::string("synthetic"));
  CHECK_EQ(doc.at("dataset").at("base_vectors_used").as_int(), static_cast<int64_t>(kUsed));
  CHECK(doc.at("dataset").at("base").at("in_manifest").as_bool());
  CHECK_EQ(doc.at("config").at("system").as_string(), system);
  CHECK_EQ(doc.at("config").at("M").as_int(), 16);
  CHECK_EQ(doc.at("config").at("ef_construction").as_int(), 200);
  CHECK_EQ(doc.at("config").at("seed").as_int(), 100);
  check_samples(doc.at("build_seconds"), 2);
  const Value& mem = doc.at("memory");
  CHECK(mem.at("vm_hwm_after_load_kib").as_int() > 0);
  CHECK(mem.at("vm_hwm_end_kib").as_int() >= mem.at("vm_hwm_after_load_kib").as_int());
  CHECK_EQ(doc.at("index").at("bytes").as_int(), static_cast<int64_t>(fs::file_size(index)));
}

// Distances recomputed here; both systems must return rows ascending by
// distance (up to kernel rounding), with k distinct valid ids.
void check_row_format(const hnsw::IvecsData& got, const std::vector<float>& base,
                      const std::vector<float>& queries, const std::string& what) {
  CHECK_EQ(got.count, kNq);
  CHECK_EQ(got.dim, kK);
  const hnsw::DistFn dist = hnsw::best_l2_squared();
  size_t bad = 0;
  for (size_t q = 0; q < kNq; ++q) {
    std::set<int32_t> seen;
    float prev = -1.0f;
    bool ok = true;
    for (size_t j = 0; j < kK && ok; ++j) {
      const int32_t id = got.data[q * kK + j];
      if (id < 0 || static_cast<size_t>(id) >= kUsed || !seen.insert(id).second) {
        ok = false;
        break;
      }
      const float d = dist(queries.data() + q * kDim, base.data() + id * kDim, kDim);
      if (d < prev - 1e-4f * prev) ok = false;
      prev = d;
    }
    if (!ok) ++bad;
  }
  if (bad) std::fprintf(stderr, "%s: %zu malformed rows\n", what.c_str(), bad);
  CHECK_EQ(bad, size_t{0});
}

void check_search(const std::string& path, const std::string& ids_dir,
                  const std::vector<float>& base, const std::vector<float>& queries) {
  const Value doc = bench::json::read_file(path);
  CHECK(has_keys(doc, {"tool", "environment", "dataset", "config", "systems", "results",
                       "sweep_seconds"}));
  check_environment(doc.at("environment"));
  CHECK_EQ(doc.at("dataset").at("name").as_string(), std::string("synthetic"));
  CHECK(!doc.at("dataset").at("ground_truth").at("in_manifest").as_bool());
  CHECK(has_keys(doc.at("systems"), {"ours", "hnswlib"}));
  CHECK_EQ(doc.at("systems").at("ours").at("load_mode").as_string(), std::string("mmap"));
  const Value& results = doc.at("results");
  CHECK_EQ(results.size(), kSystems.size() * kEfs.size() * kThreads.size());

  // Exact top-k over the indexed prefix, computed here.
  std::vector<std::set<uint32_t>> truth(kNq);
  for (size_t q = 0; q < kNq; ++q) {
    const hnsw::KnnResult r =
        hnsw::brute_force_knn(base.data(), kUsed, kDim, queries.data() + q * kDim, kK);
    truth[q].insert(r.ids.begin(), r.ids.end());
  }

  for (const Value& e : results.items()) {
    CHECK(has_keys(e, {"system", "ef", "threads", "k", "queries", "warmup_seconds", "seconds",
                       "qps_median", "recall_at_k", "matches_single_thread", "block_seconds",
                       "pass_order"}));
    const std::string system = e.at("system").as_string();
    const size_t ef = static_cast<size_t>(e.at("ef").as_int());
    const size_t threads = static_cast<size_t>(e.at("threads").as_int());
    check_samples(e.at("seconds"), kReps);
    CHECK_NEAR(e.at("qps_median").as_double(),
               static_cast<double>(kNq) / e.at("seconds").at("median").as_double(), 1e-6);
    // A B B A order across repetitions.
    const Value& order = e.at("pass_order");
    CHECK_EQ(order.size(), kReps * kSystems.size());
    CHECK_EQ(order.at(0).as_string(), std::string("ours"));
    CHECK_EQ(order.at(1).as_string(), std::string("hnswlib"));
    CHECK_EQ(order.at(2).as_string(), std::string("hnswlib"));
    CHECK_EQ(order.at(3).as_string(), std::string("ours"));

    const std::string name =
        system + "_ef" + std::to_string(ef) + "_t" + std::to_string(threads) + ".ivecs";
    const hnsw::IvecsData got = hnsw::read_ivecs((fs::path(ids_dir) / name).string());
    check_row_format(got, base, queries, name);

    // Recall recomputed independently of the harness.
    size_t hits = 0;
    for (size_t q = 0; q < kNq; ++q) {
      for (size_t j = 0; j < kK; ++j) {
        const int32_t id = got.data[q * kK + j];
        if (id >= 0 && truth[q].count(static_cast<uint32_t>(id))) ++hits;
      }
    }
    const double recall = static_cast<double>(hits) / static_cast<double>(kNq * kK);
    CHECK_NEAR(e.at("recall_at_k").as_double(), recall, 1e-12);
    if (ef == 400) {
      if (recall < 0.99) std::fprintf(stderr, "%s recall %.4f at ef 400\n", system.c_str(), recall);
      CHECK(recall >= 0.99);
    }

    if (threads == 1) {
      CHECK(e.at("matches_single_thread").is_null());
    } else {
      CHECK(e.at("matches_single_thread").as_bool());
      const std::string st = system + "_ef" + std::to_string(ef) + "_t1.ivecs";
      const hnsw::IvecsData single = hnsw::read_ivecs((fs::path(ids_dir) / st).string());
      CHECK(single.data == got.data);
    }
  }
}

void check_loadtime(const std::string& path) {
  const Value doc = bench::json::read_file(path);
  CHECK(has_keys(doc, {"tool", "environment", "dataset", "config", "results"}));
  check_environment(doc.at("environment"));
  CHECK_EQ(doc.at("dataset").at("name").as_string(), std::string("synthetic"));
  CHECK(doc.at("config").at("pre_load_command").is_null());
  const Value& results = doc.at("results");
  CHECK_EQ(results.size(), size_t{5});
  std::set<std::string> names;
  for (const Value& e : results.items()) {
    names.insert(e.at("strategy").as_string());
    check_samples(e.at("load_seconds"), 2);
    check_samples(e.at("first_queries_seconds"), 2);
  }
  CHECK_EQ(names, (std::set<std::string>{"ours-mmap-shallow", "ours-mmap-deep",
                                         "ours-read-shallow", "ours-read-deep",
                                         "hnswlib-load"}));
}

void smoke() {
  TempDir tmp;
  const std::string data = tmp / "data";
  const std::string base_file = data + "/synthetic_base.fvecs";
  const std::string query_file = data + "/synthetic_query.fvecs";
  const std::string gt_full = data + "/synthetic_groundtruth.ivecs";
  const std::string gt_used = tmp / "gt_2500.ivecs";

  if (!run("bench_make_synthetic",
           {"--out", data, "--n", std::to_string(kN), "--nq", std::to_string(kNq), "--dim",
            std::to_string(kDim), "--k", "100", "--clusters", "16", "--seed", "7",
            "--threads", "2"})) {
    return;
  }
  if (!run("bench_gt", {"check", "--base", base_file, "--queries", query_file, "--gt", gt_full,
                        "--n", std::to_string(kN), "--nq", std::to_string(kNq), "--dim",
                        std::to_string(kDim), "--k", "100"})) {
    return;
  }
  const std::string verify_json = tmp / "verify.json";
  if (!run("bench_gt", {"verify", "--base", base_file, "--queries", query_file, "--gt", gt_full,
                        "--sample", "20", "--threads", "2", "--json", verify_json})) {
    return;
  }
  const Value verify = bench::json::read_file(verify_json);
  CHECK(verify.at("result").at("ok").as_bool());
  CHECK_EQ(verify.at("result").at("mismatched_ranks").as_int(), 0);
  CHECK_EQ(verify.at("dataset").at("name").as_string(), std::string("synthetic"));

  if (!run("bench_gt", {"subset", "--base", base_file, "--queries", query_file, "--n",
                        std::to_string(kUsed), "--k", "100", "--out", gt_used, "--threads",
                        "2"})) {
    return;
  }

  const std::string ours_idx = tmp / "ours.hnsw", hnswlib_idx = tmp / "hnswlib.idx";
  for (const std::string& system : kSystems) {
    const std::string idx = system == "ours" ? ours_idx : hnswlib_idx;
    const std::string json = tmp / ("build_" + system + ".json");
    if (!run("bench_build", {"--system", system, "--data", base_file, "--n",
                             std::to_string(kUsed), "--runs", "2", "--out", idx, "--json",
                             json})) {
      return;
    }
    check_build(json, system, idx);
  }

  const std::string search_json = tmp / "search.json";
  const std::string ids_dir = tmp / "ids";
  if (!run("bench_search", {"--ours", ours_idx, "--hnswlib", hnswlib_idx, "--queries",
                            query_file, "--gt", gt_used, "--efs", "10,40,400", "--threads",
                            "1,3", "--reps", std::to_string(kReps), "--dump-ids", ids_dir,
                            "--json", search_json})) {
    return;
  }
  const hnsw::FvecsData base = hnsw::read_fvecs(base_file);
  const hnsw::FvecsData queries = hnsw::read_fvecs(query_file);
  check_search(search_json, ids_dir, base.data, queries.data);

  const std::string load_json = tmp / "load.json";
  if (!run("bench_loadtime", {"--ours", ours_idx, "--hnswlib", hnswlib_idx, "--queries",
                              query_file, "--reps", "2", "--first", "50", "--json",
                              load_json})) {
    return;
  }
  check_loadtime(load_json);

  // A ground truth that does not belong to the index is refused.
  CHECK(std::system((bench::shell_quote((fs::path(bin_dir) / "bench_search").string()) +
                     " --ours " + bench::shell_quote(ours_idx) + " --systems ours --queries " +
                     bench::shell_quote(query_file) + " --gt " + bench::shell_quote(gt_full) +
                     " --efs 10 --threads 1 --reps 1 --json " +
                     bench::shell_quote(tmp / "refused.json") + " 2>/dev/null")
                        .c_str()) != 0);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: test_bench_smoke BENCH_BIN_DIR\n");
    return 2;
  }
  bin_dir = argv[1];
  smoke();
  return test_finish();
}
