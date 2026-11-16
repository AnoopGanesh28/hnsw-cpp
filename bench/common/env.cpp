#include "env.h"

#include <sys/utsname.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#include "build_info.h"
#include "git_info.h"
#include "hnsw/config.h"
#include "hnsw/distance.h"
#include "hnswlib_index.h"
#include "proc.h"

namespace bench {

namespace {

std::string trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t");
  if (b == std::string::npos) return "";
  const size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string read_text(const std::string& path) {
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

json::Value cpu_record() {
  std::ifstream in("/proc/cpuinfo");
  std::string line, model, physical;
  size_t logical = 0;
  std::set<std::pair<std::string, std::string>> cores;
  std::set<std::string> sockets;
  while (std::getline(in, line)) {
    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = trim(line.substr(0, colon));
    const std::string value = trim(line.substr(colon + 1));
    if (key == "processor") ++logical;
    if (key == "model name" && model.empty()) model = value;
    if (key == "physical id") { physical = value; sockets.insert(value); }
    if (key == "core id") cores.emplace(physical, value);
  }
  json::Value out = json::Value::object();
  out.set("model", model);
  out.set("logical_cpus", logical);
  out.set("physical_cores", cores.size());
  out.set("sockets", sockets.size());
  out.set("hardware_concurrency", std::thread::hardware_concurrency());
  return out;
}

json::Value kernel_record() {
  json::Value out = json::Value::object();
  struct utsname u {};
  if (::uname(&u) == 0) {
    out.set("sysname", u.sysname);
    out.set("release", u.release);
    out.set("version", u.version);
    out.set("machine", u.machine);
  }
  return out;
}

json::Value wsl_record() {
  const std::string version = trim(read_text("/proc/version"));
  const char* distro = std::getenv("WSL_DISTRO_NAME");
  const bool in_version = lower(version).find("microsoft") != std::string::npos;
  json::Value out = json::Value::object();
  out.set("detected", in_version || distro != nullptr);
  out.set("proc_version", version);
  out.set("WSL_DISTRO_NAME", distro ? json::Value(distro) : json::Value());
  return out;
}

json::Value lscpu_head(size_t lines) {
  json::Value out = json::Value::array();
  try {
    std::istringstream ss(run_command("lscpu"));
    std::string line;
    while (out.size() < lines && std::getline(ss, line)) out.push(line);
  } catch (const std::exception& e) {
    out.push(std::string("unavailable: ") + e.what());
  }
  return out;
}

json::Value predefined_macros() {
  json::Value out = json::Value::object();
#ifdef __OPTIMIZE__
  out.set("__OPTIMIZE__", true);
#else
  out.set("__OPTIMIZE__", false);
#endif
#ifdef NDEBUG
  out.set("NDEBUG", true);
#else
  out.set("NDEBUG", false);
#endif
#ifdef __AVX2__
  out.set("__AVX2__", true);
#else
  out.set("__AVX2__", false);
#endif
#ifdef __FMA__
  out.set("__FMA__", true);
#else
  out.set("__FMA__", false);
#endif
#ifdef __AVX512F__
  out.set("__AVX512F__", true);
#else
  out.set("__AVX512F__", false);
#endif
  return out;
}

std::string utc_now() {
  const std::time_t t = std::time(nullptr);
  std::tm tm {};
  ::gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

}  // namespace

bool is_release_build() {
#if defined(__OPTIMIZE__) && defined(NDEBUG)
  return std::string(HNSW_BENCH_SANITIZE) == "off";
#else
  return false;
#endif
}

json::Value environment_record(size_t dim) {
  json::Value compiler = json::Value::object();
  compiler.set("path", HNSW_BENCH_CXX_COMPILER);
  compiler.set("id", HNSW_BENCH_CXX_COMPILER_ID);
  compiler.set("version", HNSW_BENCH_CXX_COMPILER_VERSION);
  compiler.set("__VERSION__", __VERSION__);
  compiler.set("gxx_version_line", HNSW_BENCH_GXX_VERSION);
  compiler.set("clangxx_version_line", HNSW_BENCH_CLANGXX_VERSION);

  json::Value per_file = json::Value::object();
  per_file.set("src/distance_avx2.cpp", HNSW_BENCH_AVX2_FILE_OPTIONS);
  per_file.set("src/distance.cpp", HNSW_BENCH_SCALAR_FILE_OPTIONS);

  json::Value build = json::Value::object();
  build.set("build_type", HNSW_BENCH_BUILD_TYPE);
  build.set("cxx_flags", HNSW_BENCH_CXX_FLAGS);
  build.set("per_file_options", per_file);
  build.set("sanitize", HNSW_BENCH_SANITIZE);
  build.set("predefined_macros", predefined_macros());
  build.set("release_build", is_release_build());

  json::Value source = json::Value::object();
  source.set("commit", HNSW_BENCH_GIT_COMMIT);
  source.set("describe", HNSW_BENCH_GIT_DESCRIBE);
  source.set("dirty", std::string(HNSW_BENCH_GIT_DIRTY) == "true");
  source.set("note", "git state of the source tree when this binary was built");

  json::Value ours = json::Value::object();
  ours.set("cpu_supports_avx2_fma", hnsw::cpu_supports_avx2_fma());
  ours.set("l2_function", hnsw::cpu_supports_avx2_fma() ? "l2_squared_avx2 (AVX2+FMA)"
                                                         : "l2_squared_scalar");
  ours.set("HNSW_OPT_EPOCH_VISITED", hnsw::kEpochVisited);
  ours.set("HNSW_OPT_ALLOC_FREE_HEAPS", hnsw::kAllocFreeHeaps);
  ours.set("HNSW_OPT_PREFETCH", hnsw::kPrefetch);

  json::Value hnswlib = hnswlib_info(dim);
  hnswlib.set("tag_pinned", HNSW_BENCH_HNSWLIB_TAG);
  hnswlib.set("commit_pinned", HNSW_BENCH_HNSWLIB_COMMIT_PINNED);
  hnswlib.set("commit_at_build", HNSW_BENCH_HNSWLIB_COMMIT);

  json::Value mem_total;
  try {
    mem_total = json::Value(read_status_kib("/proc/meminfo", "MemTotal"));
  } catch (const std::exception& e) {
    mem_total = json::Value(std::string("unavailable: ") + e.what());
  }

  json::Value out = json::Value::object();
  out.set("date_utc", utc_now());
  out.set("cpu", cpu_record());
  out.set("mem_total_kib", mem_total);
  out.set("kernel", kernel_record());
  out.set("wsl", wsl_record());
  out.set("lscpu_head", lscpu_head(20));
  out.set("compiler", compiler);
  out.set("build", build);
  out.set("source", source);
  out.set("ours", ours);
  out.set("hnswlib", hnswlib);
  return out;
}

}  // namespace bench
