// Layout ablation arm: the M2 scalar baseline (tag m2-scalar-baseline, before
// the contiguous per-node layout) built and searched through its public API.
// It is compiled against a worktree of that tag by bench/m2_baseline/build.sh,
// not against the current library, so it shares no code with bench/common and
// writes its JSON by hand.
//
//   bench_m2 --base BASE.fvecs --json OUT.json [--n all] [--M 16] [--efc 200]
//            [--seed 100] [--build-only]
//            [--queries Q.fvecs --gt GT.ivecs --efs E1,E2 --k 10 --reps 7
//             --between CMD]
//
// The index is built once (M2 has no file format). Then, for r = 0 .. reps-1:
// CMD runs through /bin/sh with argument r (outside any timing), and every ef
// gets one untimed warm-up pass and one timed pass over all queries; CMD runs
// once more with argument reps at the end. With CMD measuring the comparison
// arm, every M2 repetition is bracketed by a measurement just before and just
// after it. A pass is a plain single-threaded loop; M2's search() returns a
// fresh KnnResult per query, which is part of what M2 costs. Results are
// padded to k with 0xFFFFFFFF and hashed (64-bit FNV-1a over the id array) the
// same way bench_search hashes them, so the two arms can be checked for
// identical results.
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw/io.h"

#ifndef BENCH_M2_FLAGS
#error "BENCH_M2_FLAGS (the compile flags) is defined by bench/m2_baseline/build.sh"
#endif
#ifndef BENCH_M2_COMMIT
#error "BENCH_M2_COMMIT (the baseline commit) is defined by bench/m2_baseline/build.sh"
#endif

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t kPad = 0xFFFFFFFFu;

double since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::string fnv1a64_hex(const void* data, size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  char out[17];
  std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(h));
  return out;
}

std::string quote(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (c == '\n') {
      out += "\\n";
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

std::string num(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

std::string list(const std::vector<double>& v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + num(v[i]);
  return out + "]";
}

std::vector<size_t> parse_sizes(const std::string& s) {
  std::vector<size_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoul(item));
  return out;
}

std::string affinity() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) return "\"unavailable\"";
  std::string out = "[";
  bool first = true;
  for (int c = 0; c < CPU_SETSIZE; ++c) {
    if (!CPU_ISSET(c, &set)) continue;
    out += (first ? "" : ", ") + std::to_string(c);
    first = false;
  }
  return out + "]";
}

std::string utc_now() {
  const std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

int run(int argc, char** argv) {
  std::map<std::string, std::string> args;
  bool build_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--build-only") {
      build_only = true;
      continue;
    }
    if (a.rfind("--", 0) != 0 || i + 1 >= argc) throw std::invalid_argument("bad argument " + a);
    args[a.substr(2)] = argv[++i];
  }
  const std::set<std::string> known = {"base", "json", "n",    "M",    "efc",  "seed",
                                       "queries", "gt", "efs", "k", "reps", "between"};
  for (const auto& kv : args) {
    if (!known.count(kv.first)) throw std::invalid_argument("unknown flag --" + kv.first);
  }
  auto get = [&](const std::string& key, const std::string& def) {
    const auto it = args.find(key);
    if (it != args.end()) return it->second;
    if (def.empty()) throw std::invalid_argument("missing --" + key);
    return def;
  };
  const std::string base_path = get("base", ""), json_path = get("json", "");
  hnsw::HnswParams params;
  params.M = std::stoul(get("M", "16"));
  params.ef_construction = std::stoul(get("efc", "200"));
  params.seed = std::stoull(get("seed", "100"));

  const hnsw::FvecsData base = hnsw::read_fvecs(base_path);
  const size_t n = get("n", "all") == "all" ? base.count : std::stoul(get("n", "all"));
  if (n == 0 || n > base.count) throw std::invalid_argument("--n out of range");
  const size_t dim = base.dim;

  const auto t0 = Clock::now();
  hnsw::HnswIndex index(dim, params);
  for (size_t i = 0; i < n; ++i) index.add(base.data.data() + i * dim);
  const double build_seconds = since(t0);
  std::fprintf(stderr, "bench_m2: built %zu vectors in %.1f s\n", n, build_seconds);

  std::ostringstream doc;
  doc << "{\n  \"tool\": \"bench_m2\",\n  \"environment\": {\"date_utc\": " << quote(utc_now())
      << ", \"compiler\": " << quote(__VERSION__) << ", \"cxx_flags\": " << quote(BENCH_M2_FLAGS)
      << ", \"m2_commit\": " << quote(BENCH_M2_COMMIT) << ", \"cpu_affinity\": " << affinity()
      << "},\n  \"config\": {\"base\": " << quote(base_path) << ", \"n\": " << n
      << ", \"dim\": " << dim << ", \"M\": " << params.M
      << ", \"ef_construction\": " << params.ef_construction << ", \"seed\": " << params.seed
      << ", \"kernel\": \"l2_squared_scalar\"},\n  \"build_seconds\": " << num(build_seconds);

  if (!build_only) {
    const hnsw::FvecsData queries = hnsw::read_fvecs(get("queries", ""));
    const hnsw::IvecsData gt = hnsw::read_ivecs(get("gt", ""));
    const std::vector<size_t> efs = parse_sizes(get("efs", ""));
    const size_t k = std::stoul(get("k", "10"));
    const size_t reps = std::stoul(get("reps", "7"));
    const std::string between = get("between", "");
    const size_t nq = queries.count;
    if (queries.dim != dim || gt.count < nq || gt.dim < k || efs.empty() || reps == 0 || k == 0) {
      throw std::invalid_argument("queries, ground truth, --efs, --k or --reps do not fit");
    }
    std::vector<uint32_t> ids(nq * k), warm(nq * k);
    auto pass = [&](size_t ef, std::vector<uint32_t>& out) {
      const auto t = Clock::now();
      for (size_t q = 0; q < nq; ++q) {
        const hnsw::KnnResult r = index.search(queries.data.data() + q * dim, k, ef);
        uint32_t* row = out.data() + q * k;
        std::copy(r.ids.begin(), r.ids.end(), row);
        std::fill(row + r.ids.size(), row + k, kPad);
      }
      return since(t);
    };
    auto hook = [&](size_t i) {
      const std::string cmd = between + " " + std::to_string(i);
      if (std::system(cmd.c_str()) != 0) throw std::runtime_error("--between failed: " + cmd);
    };
    std::map<size_t, std::vector<double>> times, warmups;
    std::map<size_t, std::vector<uint32_t>> result_ids;  // identical across repetitions
    for (size_t r = 0; r < reps; ++r) {
      hook(r);
      for (size_t ef : efs) {
        warmups[ef].push_back(pass(ef, warm));
        times[ef].push_back(pass(ef, ids));
        if (ids != warm) throw std::runtime_error("timed pass differs from warm-up pass");
        if (r > 0 && ids != result_ids[ef]) throw std::runtime_error("results changed between reps");
        result_ids[ef] = ids;
        std::fprintf(stderr, "bench_m2: rep %zu ef %zu: %.3f s\n", r, ef, times[ef].back());
      }
    }
    hook(reps);

    doc << ",\n  \"reps\": " << reps << ",\n  \"queries\": " << nq << ",\n  \"results\": [";
    for (size_t i = 0; i < efs.size(); ++i) {
      const size_t ef = efs[i];
      const std::vector<uint32_t>& got = result_ids[ef];
      size_t hits = 0;
      for (size_t q = 0; q < nq; ++q) {
        std::set<uint32_t> truth;
        for (size_t j = 0; j < k; ++j) truth.insert(static_cast<uint32_t>(gt.data[q * gt.dim + j]));
        for (size_t j = 0; j < k; ++j) hits += truth.count(got[q * k + j]);
      }
      const std::vector<double>& s = times[ef];
      const double med = median(s);
      doc << (i ? "," : "") << "\n    {\"system\": \"m2\", \"ef\": " << ef << ", \"k\": " << k
          << ", \"threads\": 1, \"warmup_seconds\": " << list(warmups[ef])
          << ", \"seconds\": {\"raw\": " << list(s) << ", \"median\": " << num(med)
          << ", \"min\": " << num(*std::min_element(s.begin(), s.end()))
          << ", \"max\": " << num(*std::max_element(s.begin(), s.end()))
          << "}, \"qps_median\": " << num(static_cast<double>(nq) / med)
          << ", \"recall_at_k\": " << num(static_cast<double>(hits) / static_cast<double>(nq * k))
          << ", \"ids_hash\": " << quote(fnv1a64_hex(got.data(), got.size() * sizeof(uint32_t)))
          << "}";
    }
    doc << "\n  ],\n  \"between\": " << quote(between);
  }
  doc << "\n}\n";
  std::ofstream out(json_path);
  out << doc.str();
  if (!out.flush()) throw std::runtime_error("cannot write " + json_path);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_m2: error: %s\n", e.what());
    return 1;
  }
}
