// Recall/QPS sweep over efSearch and thread counts for ours and hnswlib.
//
//   bench_search --queries Q.fvecs --gt GT.ivecs --json OUT.json
//                [--ours INDEX] [--hnswlib INDEX] [--systems ours,hnswlib]
//                [--efs 10,12,...,800] [--threads 1,4,8] [--reps 5] [--k 10]
//                [--nq all] [--chunk 8] [--load-mode mmap|read]
//                [--validate deep|shallow] [--dump-ids DIR]
//
// Method, for each thread count T, then each ef:
//   - hnswlib's setEf is called first (it is not thread-safe);
//   - one untimed warm-up pass over all queries per system;
//   - `reps` timed passes per system, in A B B A A B ... order across
//     repetitions so slow drift (thermal, frequency) affects both equally;
//   - every raw pass time is stored, plus median / min / max.
// A pass times only the query loop: results go to preallocated arrays and
// nothing else happens inside the timed region. T = 1 is a plain loop on the
// calling thread; T > 1 runs the SAME dispatch code for both systems on one
// ThreadPool (created before timing), chunk queries per work item, timed from
// submission to completion. Each pass's results must equal the warm-up's
// (checked after timing), and recall@k is computed afterwards, by one function
// for both systems, against the first k ground-truth ids.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "args.h"
#include "dataset.h"
#include "env.h"
#include "hnsw/index_file.h"
#include "hnsw/io.h"
#include "hnsw/thread_pool.h"
#include "hnswlib_index.h"
#include "json.h"
#include "stats.h"
#include "tool.h"

namespace {

using bench::json::Value;

const std::vector<size_t> kDefaultEfs = {10,  12,  16,  20,  24,  32,  40,  48,  64,  80,
                                         96,  128, 160, 200, 256, 320, 400, 512, 640, 800};
const std::vector<size_t> kDefaultThreads = {1, 4, 8};

// One scratch per thread index, each on its own cache lines.
struct alignas(64) PaddedScratch {
  hnsw::SearchScratch scratch;
};

// Mean over queries of |first k found ids ∩ first k ground-truth ids| / k.
// The one recall function used for both systems.
double mean_recall(const std::vector<uint32_t>& ids, size_t nq, size_t k,
                   const hnsw::IvecsData& gt) {
  double sum = 0.0;
  std::vector<uint32_t> found(k), truth(k);
  for (size_t q = 0; q < nq; ++q) {
    std::copy(ids.begin() + q * k, ids.begin() + (q + 1) * k, found.begin());
    for (size_t j = 0; j < k; ++j) truth[j] = static_cast<uint32_t>(gt.data[q * gt.dim + j]);
    sum += hnsw::recall_at_k(found, truth, k);
  }
  return sum / static_cast<double>(nq);
}

// One pass over all queries. search_one(thread_index, q, ids, dists) writes
// query q's results and returns how many; rows are padded to k here.
template <typename SearchOne>
double timed_pass(size_t nq, size_t k, hnsw::ThreadPool* pool, size_t chunk,
                  const SearchOne& search_one, std::vector<uint32_t>& ids,
                  std::vector<float>& dists) {
  uint32_t* const out_ids = ids.data();
  float* const out_dists = dists.data();
  auto run_query = [&](size_t thread, size_t q) {
    uint32_t* row_ids = out_ids + q * k;
    float* row_dists = out_dists + q * k;
    const size_t got = search_one(thread, q, row_ids, row_dists);
    std::fill(row_ids + got, row_ids + k, hnsw::kNoNeighbor);
    std::fill(row_dists + got, row_dists + k, std::numeric_limits<float>::infinity());
  };
  if (!pool) {
    const auto t0 = bench::Clock::now();
    for (size_t q = 0; q < nq; ++q) run_query(0, q);
    return bench::seconds_between(t0, bench::Clock::now());
  }
  const hnsw::ThreadPool::Task task = [&](size_t thread, size_t begin, size_t end) {
    for (size_t q = begin; q < end; ++q) run_query(thread, q);
  };
  const auto t0 = bench::Clock::now();
  pool->parallel_for(nq, chunk, task);
  return bench::seconds_between(t0, bench::Clock::now());
}

void dump_ids(const std::string& dir, const std::string& system, size_t ef, size_t threads,
              const std::vector<uint32_t>& ids, size_t k) {
  std::vector<int32_t> out(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    out[i] = ids[i] == hnsw::kNoNeighbor ? -1 : static_cast<int32_t>(ids[i]);
  }
  const std::string name = system + "_ef" + std::to_string(ef) + "_t" +
                           std::to_string(threads) + ".ivecs";
  hnsw::write_ivecs((std::filesystem::path(dir) / name).string(), out, k);
}

int run(int argc, char** argv) {
  bench::Args args(argc, argv);
  const std::string query_path = args.str("queries");
  const std::string gt_path = args.str("gt");
  const std::string json_out = args.str("json");
  const std::vector<std::string> systems = args.words("systems", {"ours", "hnswlib"});
  const std::vector<size_t> efs = args.sizes("efs", kDefaultEfs);
  const std::vector<size_t> thread_counts = args.sizes("threads", kDefaultThreads);
  const size_t reps = args.size("reps", 5);
  const size_t k = args.size("k", 10);
  const size_t nq_flag = args.size("nq", 0);
  const size_t chunk = args.size("chunk", hnsw::kDefaultBatchChunk);
  const std::string load_mode = args.choice("load-mode", {"mmap", "read"}, "mmap");
  const std::string validate = args.choice("validate", {"deep", "shallow"}, "deep");
  const std::string dump_dir = args.str("dump-ids", "");
  bool use_ours = false, use_hnswlib = false;
  for (const std::string& s : systems) {
    if (s == "ours") use_ours = true;
    else if (s == "hnswlib") use_hnswlib = true;
    else throw std::invalid_argument("--systems: unknown system '" + s + "'");
  }
  if (systems.size() != static_cast<size_t>(use_ours) + static_cast<size_t>(use_hnswlib)) {
    throw std::invalid_argument("--systems lists a system twice");
  }
  const std::string ours_path = use_ours ? args.str("ours") : args.str("ours", "");
  const std::string hnswlib_path = use_hnswlib ? args.str("hnswlib") : args.str("hnswlib", "");
  args.finish();
  if (reps == 0 || k == 0 || chunk == 0) {
    throw std::invalid_argument("--reps, --k and --chunk must be >= 1");
  }
  for (size_t t : thread_counts) {
    if (t == 0) throw std::invalid_argument("--threads entries must be >= 1");
  }

  const bench::Manifest manifest = bench::manifest_for(query_path);
  const hnsw::FvecsData queries = hnsw::read_fvecs(query_path);
  const hnsw::IvecsData gt = hnsw::read_ivecs(gt_path);
  const size_t nq = nq_flag == 0 ? queries.count : nq_flag;
  const size_t dim = queries.dim;
  if (nq > queries.count || nq > gt.count) {
    throw std::invalid_argument("--nq exceeds the number of queries or ground-truth rows");
  }
  if (gt.dim < k) throw std::invalid_argument("ground truth rows are shorter than k");

  // Load both systems up front; the sweep interleaves them.
  Value systems_info = Value::object();
  std::optional<hnsw::LoadedIndex> ours;
  std::unique_ptr<bench::HnswlibIndex> hnswlib;
  size_t index_size = 0;
  if (use_ours) {
    hnsw::LoadOptions options;
    options.mode = load_mode == "mmap" ? hnsw::LoadMode::Mmap : hnsw::LoadMode::Read;
    options.validate_deep = validate == "deep";
    const auto t0 = bench::Clock::now();
    ours.emplace(hnsw::LoadedIndex::open(ours_path, options));
    const double secs = bench::seconds_between(t0, bench::Clock::now());
    if (ours->dim() != dim) throw std::runtime_error("ours: index and query dimensions differ");
    index_size = ours->size();
    Value info = Value::object();
    info.set("index", std::filesystem::absolute(ours_path).string());
    info.set("bytes", ours->file_size());
    info.set("load_mode", load_mode);
    info.set("validate_deep", options.validate_deep);
    info.set("load_seconds", secs);
    info.set("size", ours->size());
    info.set("M", ours->M());
    info.set("ef_construction", ours->ef_construction());
    systems_info.set("ours", info);
  }
  if (use_hnswlib) {
    const auto t0 = bench::Clock::now();
    hnswlib = bench::HnswlibIndex::load(hnswlib_path, dim);
    const double secs = bench::seconds_between(t0, bench::Clock::now());
    if (use_ours && hnswlib->size() != index_size) {
      throw std::runtime_error("the two indexes hold different numbers of vectors");
    }
    index_size = hnswlib->size();
    Value info = Value::object();
    info.set("index", std::filesystem::absolute(hnswlib_path).string());
    info.set("bytes", std::filesystem::file_size(hnswlib_path));
    info.set("load_mode", "hnswlib loadIndex (read into memory)");
    info.set("load_seconds", secs);
    info.set("size", hnswlib->size());
    systems_info.set("hnswlib", info);
  }
  for (size_t i = 0; i < nq * gt.dim; ++i) {
    if (gt.data[i] < 0 || static_cast<size_t>(gt.data[i]) >= index_size) {
      throw std::runtime_error("ground-truth id " + std::to_string(gt.data[i]) +
                               " is not in the index (" + std::to_string(index_size) +
                               " vectors): wrong ground truth for this index?");
    }
  }
  if (!dump_dir.empty()) std::filesystem::create_directories(dump_dir);

  const size_t max_threads = *std::max_element(thread_counts.begin(), thread_counts.end());
  std::vector<PaddedScratch> scratch(max_threads);
  const float* const qdata = queries.data.data();
  size_t ef = 0;  // the current sweep point, read by the search lambdas
  auto search_ours = [&](size_t thread, size_t q, uint32_t* ids, float* dists) {
    return ours->search_into(qdata + q * dim, k, ef, scratch[thread].scratch, ids, dists);
  };
  auto search_hnswlib = [&](size_t, size_t q, uint32_t* ids, float* dists) {
    return hnswlib->search_into(qdata + q * dim, k, ids, dists);
  };

  struct Buffers {
    std::vector<uint32_t> ids, warm_ids;
    std::vector<float> dists, warm_dists;
  };
  std::map<std::string, Buffers> buf;
  for (const std::string& s : systems) {
    buf[s] = {std::vector<uint32_t>(nq * k), std::vector<uint32_t>(nq * k),
              std::vector<float>(nq * k), std::vector<float>(nq * k)};
  }
  // Single-thread results per (system, ef), to check multi-thread runs against.
  std::map<std::pair<std::string, size_t>, std::vector<uint32_t>> single_thread_ids;

  Value results = Value::array();
  const auto sweep_start = bench::Clock::now();
  for (const size_t threads : thread_counts) {
    std::unique_ptr<hnsw::ThreadPool> pool;
    if (threads > 1) pool = std::make_unique<hnsw::ThreadPool>(threads);
    for (const size_t ef_value : efs) {
      ef = ef_value;
      const auto block_start = bench::Clock::now();
      if (hnswlib) hnswlib->set_ef(ef);
      auto pass = [&](const std::string& s, std::vector<uint32_t>& ids,
                      std::vector<float>& dists) {
        return s == "ours" ? timed_pass(nq, k, pool.get(), chunk, search_ours, ids, dists)
                           : timed_pass(nq, k, pool.get(), chunk, search_hnswlib, ids, dists);
      };
      std::map<std::string, double> warmup;
      for (const std::string& s : systems) warmup[s] = pass(s, buf[s].warm_ids, buf[s].warm_dists);

      std::map<std::string, std::vector<double>> times;
      Value order = Value::array();
      for (size_t r = 0; r < reps; ++r) {
        std::vector<std::string> seq = systems;
        if (r % 2 == 1) std::reverse(seq.begin(), seq.end());
        for (const std::string& s : seq) {
          times[s].push_back(pass(s, buf[s].ids, buf[s].dists));
          order.push(s);
          if (buf[s].ids != buf[s].warm_ids || buf[s].dists != buf[s].warm_dists) {
            throw std::runtime_error(s + ": results of a timed pass differ from the warm-up "
                                     "pass (ef " + std::to_string(ef) + ", threads " +
                                     std::to_string(threads) + ")");
          }
        }
      }
      const double block_seconds = bench::seconds_between(block_start, bench::Clock::now());

      std::string progress;
      for (const std::string& s : systems) {
        const double recall = mean_recall(buf[s].ids, nq, k, gt);
        const bench::Summary sum = bench::summarize(times[s]);
        Value matches;
        const auto key = std::make_pair(s, ef);
        if (threads == 1) {
          single_thread_ids[key] = buf[s].ids;
        } else if (single_thread_ids.count(key)) {
          matches = Value(single_thread_ids[key] == buf[s].ids);
        }
        if (!dump_dir.empty()) dump_ids(dump_dir, s, ef, threads, buf[s].ids, k);

        Value entry = Value::object();
        entry.set("system", s);
        entry.set("ef", ef);
        entry.set("threads", threads);
        entry.set("k", k);
        entry.set("queries", nq);
        entry.set("warmup_seconds", warmup[s]);
        entry.set("seconds", bench::samples_json(times[s]));
        entry.set("qps_median", static_cast<double>(nq) / sum.median);
        entry.set("recall_at_k", recall);
        entry.set("matches_single_thread", matches);
        entry.set("block_seconds", block_seconds);
        entry.set("pass_order", order);
        results.push(entry);
        char line[160];
        std::snprintf(line, sizeof(line), " %s %.0f qps recall %.4f", s.c_str(),
                      static_cast<double>(nq) / sum.median, recall);
        progress += line;
      }
      std::fprintf(stderr, "bench_search: threads %zu ef %zu:%s (%.1f s)\n", threads, ef,
                   progress.c_str(), block_seconds);
    }
  }
  const double sweep_seconds = bench::seconds_between(sweep_start, bench::Clock::now());

  Value config = Value::object();
  config.set("systems", Value::array_of(systems));
  config.set("efs", Value::array_of(efs));
  config.set("threads", Value::array_of(thread_counts));
  config.set("reps", reps);
  config.set("warmup_passes", 1);
  config.set("k", k);
  config.set("queries", nq);
  config.set("chunk", chunk);
  config.set("rep_order", "systems in --systems order on even repetitions, reversed on odd");
  config.set("single_thread", "plain loop on the calling thread");
  config.set("multi_thread", "hnsw::ThreadPool::parallel_for, same dispatch for both systems");

  Value dataset = Value::object();
  dataset.set("name", manifest.dataset);
  dataset.set("queries", bench::file_record(query_path, manifest));
  dataset.set("ground_truth", bench::file_record(gt_path, manifest));
  dataset.set("index_vectors", index_size);

  Value doc = Value::object();
  doc.set("tool", "bench_search");
  doc.set("environment", bench::environment_record(dim));
  doc.set("dataset", dataset);
  doc.set("config", config);
  doc.set("systems", systems_info);
  doc.set("results", results);
  doc.set("sweep_seconds", sweep_seconds);
  bench::json::write_file(json_out, doc);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  return bench::run_tool("bench_search", [&] { return run(argc, argv); });
}
