// Index load time, and time to answer the first queries right after loading.
//
//   bench_loadtime --queries Q.fvecs --json OUT.json [--ours INDEX] [--hnswlib INDEX]
//                  [--strategies ours-mmap-shallow,ours-mmap-deep,ours-read-shallow,
//                                ours-read-deep,hnswlib-load]
//                  [--reps 5] [--first 1000] [--ef 100] [--k 10]
//                  [--pre-load-command CMD]
//
// Each repetition loads every strategy once (order reversed on odd
// repetitions). One measurement: [load] then [first `first` queries,
// single-threaded, ef, k, starting with a fresh SearchScratch], each timed with
// steady_clock; the index is destroyed (unmapped / freed) after timing.
//
// Without --pre-load-command the page cache is warm: every strategy is loaded
// once untimed before measuring. With it, CMD runs through /bin/sh before
// EVERY load, outside the timed region, and must exit 0; with a cache-dropping
// command (bench/run_cold_loadtime.sh) that measures cold loads. The JSON
// records which of the two was done.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "args.h"
#include "dataset.h"
#include "env.h"
#include "hnsw/index_file.h"
#include "hnsw/io.h"
#include "hnswlib_index.h"
#include "json.h"
#include "stats.h"
#include "tool.h"

namespace {

using bench::json::Value;

const std::vector<std::string> kStrategies = {"ours-mmap-shallow", "ours-mmap-deep",
                                              "ours-read-shallow", "ours-read-deep",
                                              "hnswlib-load"};

struct Measurement {
  double load_seconds = 0.0;
  double first_queries_seconds = 0.0;
};

int run(int argc, char** argv) {
  bench::Args args(argc, argv);
  const std::string query_path = args.str("queries");
  const std::string json_out = args.str("json");
  const std::vector<std::string> strategies = args.words("strategies", kStrategies);
  const size_t reps = args.size("reps", 5);
  const size_t first = args.size("first", 1000);
  const size_t ef = args.size("ef", 100);
  const size_t k = args.size("k", 10);
  const std::string pre_load = args.str("pre-load-command", "");
  bool need_ours = false, need_hnswlib = false;
  for (const std::string& s : strategies) {
    bool known = false;
    for (const std::string& t : kStrategies) known = known || s == t;
    if (!known) throw std::invalid_argument("--strategies: unknown strategy '" + s + "'");
    (s == "hnswlib-load" ? need_hnswlib : need_ours) = true;
  }
  const std::string ours_path = need_ours ? args.str("ours") : args.str("ours", "");
  const std::string hnswlib_path = need_hnswlib ? args.str("hnswlib") : args.str("hnswlib", "");
  args.finish();
  if (reps == 0 || k == 0) throw std::invalid_argument("--reps and --k must be >= 1");

  const bench::Manifest manifest = bench::manifest_for(query_path);
  const hnsw::FvecsData queries = hnsw::read_fvecs(query_path);
  if (first == 0 || first > queries.count) {
    throw std::invalid_argument("--first must be in [1, number of queries]");
  }
  const size_t dim = queries.dim;
  std::vector<uint32_t> ids(first * k);
  std::vector<float> dists(first * k);

  auto measure = [&](const std::string& s) {
    if (!pre_load.empty() && std::system(pre_load.c_str()) != 0) {
      throw std::runtime_error("--pre-load-command failed: " + pre_load);
    }
    Measurement m;
    size_t answered = 0;
    if (s == "hnswlib-load") {
      const auto t0 = bench::Clock::now();
      std::unique_ptr<bench::HnswlibIndex> index = bench::HnswlibIndex::load(hnswlib_path, dim);
      const auto t1 = bench::Clock::now();
      index->set_ef(ef);
      for (size_t q = 0; q < first; ++q) {
        answered += index->search_into(queries.data.data() + q * dim, k, ids.data() + q * k,
                                       dists.data() + q * k);
      }
      const auto t2 = bench::Clock::now();
      m.load_seconds = bench::seconds_between(t0, t1);
      m.first_queries_seconds = bench::seconds_between(t1, t2);
    } else {
      hnsw::LoadOptions options;
      options.mode = s.find("mmap") != std::string::npos ? hnsw::LoadMode::Mmap
                                                         : hnsw::LoadMode::Read;
      options.validate_deep = s.find("deep") != std::string::npos;
      const auto t0 = bench::Clock::now();
      std::optional<hnsw::LoadedIndex> index(hnsw::LoadedIndex::open(ours_path, options));
      const auto t1 = bench::Clock::now();
      hnsw::SearchScratch scratch;
      for (size_t q = 0; q < first; ++q) {
        answered += index->search_into(queries.data.data() + q * dim, k, ef, scratch,
                                       ids.data() + q * k, dists.data() + q * k);
      }
      const auto t2 = bench::Clock::now();
      m.load_seconds = bench::seconds_between(t0, t1);
      m.first_queries_seconds = bench::seconds_between(t1, t2);
    }
    if (answered != first * k) {
      throw std::runtime_error(s + ": answered " + std::to_string(answered) +
                               " results, expected " + std::to_string(first * k));
    }
    return m;
  };

  if (pre_load.empty()) {
    for (const std::string& s : strategies) measure(s);
  }
  std::vector<std::vector<double>> load(strategies.size()), query(strategies.size());
  for (size_t r = 0; r < reps; ++r) {
    for (size_t j = 0; j < strategies.size(); ++j) {
      const size_t i = r % 2 == 0 ? j : strategies.size() - 1 - j;
      const Measurement m = measure(strategies[i]);
      load[i].push_back(m.load_seconds);
      query[i].push_back(m.first_queries_seconds);
    }
  }

  Value entries = Value::array();
  for (size_t i = 0; i < strategies.size(); ++i) {
    const std::string& path = strategies[i] == "hnswlib-load" ? hnswlib_path : ours_path;
    Value e = Value::object();
    e.set("strategy", strategies[i]);
    e.set("index", std::filesystem::absolute(path).string());
    e.set("bytes", std::filesystem::file_size(path));
    e.set("load_seconds", bench::samples_json(load[i]));
    e.set("first_queries_seconds", bench::samples_json(query[i]));
    entries.push(e);
    std::fprintf(stderr, "bench_loadtime: %-18s load median %.6f s, first %zu queries %.6f s\n",
                 strategies[i].c_str(), bench::summarize(load[i]).median, first,
                 bench::summarize(query[i]).median);
  }

  Value config = Value::object();
  config.set("strategies", Value::array_of(strategies));
  config.set("reps", reps);
  config.set("first_queries", first);
  config.set("ef", ef);
  config.set("k", k);
  config.set("page_cache", pre_load.empty() ? "warm (one untimed load per strategy first)"
                                            : "pre-load command before every load");
  config.set("pre_load_command", pre_load.empty() ? Value() : Value(pre_load));

  Value dataset = Value::object();
  dataset.set("name", manifest.dataset);
  dataset.set("queries", bench::file_record(query_path, manifest));

  Value doc = Value::object();
  doc.set("tool", "bench_loadtime");
  doc.set("environment", bench::environment_record(dim));
  doc.set("dataset", dataset);
  doc.set("config", config);
  doc.set("results", entries);
  bench::json::write_file(json_out, doc);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  return bench::run_tool("bench_loadtime", [&] { return run(argc, argv); });
}
