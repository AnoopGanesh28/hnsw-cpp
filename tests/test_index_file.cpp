// Index file format: round trips in both load modes, the byte layout against
// docs/FORMAT.md, format stability, atomic save, and LoadedIndex semantics.

#include <sys/stat.h>
#include <unistd.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw/index_file.h"
#include "index_file_util.h"
#include "test_helper.h"

using namespace hnsw;
namespace fs = std::filesystem;

namespace {

// Directory removed (recursively) on destruction.
class TempDir {
 public:
  TempDir() {
    std::string tmpl = (fs::temp_directory_path() / "hnsw_test_dir_XXXXXX").string();
    path_ = ::mkdtemp(tmpl.data()) ? tmpl : std::string();
  }
  ~TempDir() {
    std::error_code ec;
    fs::permissions(path_, fs::perms::owner_all, fs::perm_options::add, ec);
    fs::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  std::string file(const std::string& name) const { return path_ + "/" + name; }
  const std::string& path() const { return path_; }
  std::vector<std::string> entries() const {
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(path_)) out.push_back(e.path().filename().string());
    return out;
  }

 private:
  std::string path_;
};

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// Graph structure, vectors, and search results (ids and distance bits) of a
// loaded index equal the original's.
void check_same(const HnswIndex& a, const LoadedIndex& b) {
  CHECK_EQ(a.size(), b.size());
  CHECK_EQ(a.dim(), b.dim());
  CHECK_EQ(a.params().M, b.M());
  CHECK_EQ(a.params().ef_construction, b.ef_construction());
  CHECK_EQ(a.max_level(), b.max_level());
  if (a.size() != b.size() || a.dim() != b.dim()) return;
  if (a.size() > 0) CHECK_EQ(a.entry_point(), b.entry_point());

  bool graph_ok = true, vectors_ok = true;
  for (uint32_t id = 0; id < a.size(); ++id) {
    graph_ok = graph_ok && a.node_level(id) == b.node_level(id);
    for (int l = 0; graph_ok && l <= a.node_level(id); ++l) {
      const NeighborView x = a.neighbors(id, l), y = b.neighbors(id, l);
      graph_ok = std::equal(x.begin(), x.end(), y.begin(), y.end());
    }
    vectors_ok = vectors_ok &&
                 std::memcmp(a.view().vector(id), b.vector(id), a.dim() * sizeof(float)) == 0;
  }
  CHECK(graph_ok);
  CHECK(vectors_ok);

  const size_t nq = 60;
  const std::vector<float> queries = test::exact_data(nq, a.dim(), 4242);
  const std::pair<size_t, size_t> kef[] = {{1, 10}, {10, 10}, {10, 64}, {a.size() + 3, 32}};
  SearchScratch sa, sb;
  bool search_ok = true;
  for (size_t q = 0; q < nq; ++q) {
    const float* qp = queries.data() + q * a.dim();
    for (const auto& [k, ef] : kef) {
      const KnnResult ra = a.search(qp, k, ef, sa), rb = b.search(qp, k, ef, sb);
      search_ok = search_ok && ra.ids == rb.ids && same_bits(ra.dists, rb.dists);
    }
  }
  CHECK(search_ok);
}

struct RoundTripConfig {
  size_t n, dim, M;
};

}  // namespace

static void test_crc_reference() {
  const char* s = "123456789";
  CHECK_EQ(test::crc32_bitwise(reinterpret_cast<const uint8_t*>(s), 9), 0xCBF43926u);
}

static void test_round_trip() {
  const RoundTripConfig configs[] = {
      {0, 7, 8},     {1, 1, 2},     {1, 128, 16},  {2, 7, 2},     {2, 32, 16},
      {300, 1, 2},   {300, 7, 8},   {400, 32, 16}, {300, 128, 2}, {2500, 7, 8},
      {2000, 32, 16}, {1500, 128, 16}};
  std::vector<DistFn> kernels{l2_squared_scalar};
  if (cpu_supports_avx2_fma()) kernels.push_back(l2_squared_avx2);
  const TempDir dir;
  bool saw_flat = false, saw_layers = false;
  for (DistFn dist : kernels) {
    for (const RoundTripConfig& c : configs) {
      const HnswIndex index = test::build_exact(c.n, c.dim, c.M, 40, c.n + c.dim + c.M, dist);
      if (c.n >= 2 && index.max_level() == 0) saw_flat = true;
      if (index.max_level() > 0) saw_layers = true;
      const std::string path = dir.file("rt.idx");
      index.save(path);
      for (LoadMode mode : {LoadMode::Mmap, LoadMode::Read}) {
        LoadOptions opt;
        opt.mode = mode;
        opt.dist = dist;
        const LoadedIndex loaded = LoadedIndex::open(path, opt);
        CHECK(loaded.mode() == mode);
        check_same(index, loaded);
      }
    }
  }
  CHECK(saw_flat);    // some multi-node config has no upper layers
  CHECK(saw_layers);  // and some has them
}

// Header fields and section contents match docs/FORMAT.md, and the arena
// section is the in-memory arena byte for byte.
static void test_byte_layout() {
  const TempDir dir;
  for (size_t n : {0u, 1u, 700u}) {
    const HnswIndex index = test::build_exact(n, 7, 8, 40, 11, l2_squared_scalar);
    const std::string path = dir.file("layout.idx");
    index.save(path);
    const std::vector<uint8_t> f = test::read_bytes(path);
    CHECK(f.size() >= test::hdr::kSize);
    CHECK(std::memcmp(f.data(), "HNSWIDX\0", 8) == 0);
    CHECK_EQ(test::get_field<uint32_t>(f, test::hdr::kVersion), 1u);
    const uint8_t marker[] = {0x04, 0x03, 0x02, 0x01};  // 0x01020304, little-endian
    CHECK(std::memcmp(f.data() + test::hdr::kEndian, marker, 4) == 0);
    CHECK_EQ(test::get_field<uint32_t>(f, test::hdr::kMetric), 0u);
    CHECK_EQ(test::get_field<uint32_t>(f, test::hdr::kDim), 7u);
    CHECK_EQ(test::get_field<uint32_t>(f, test::hdr::kM), 8u);
    CHECK_EQ(test::get_field<uint64_t>(f, test::hdr::kCount), uint64_t{n});
    CHECK_EQ(test::get_field<int32_t>(f, test::hdr::kMaxLevel), index.max_level());
    CHECK_EQ(test::get_field<uint64_t>(f, test::hdr::kStride), uint64_t{index.layout().record_stride});
    CHECK_EQ(test::get_field<uint64_t>(f, test::hdr::kFileSize), uint64_t{f.size()});
    CHECK_EQ(test::get_field<uint32_t>(f, test::hdr::kCrc),
             test::crc32_bitwise(f.data(), test::hdr::kCrc));
    bool reserved_zero = true;
    for (size_t i = test::hdr::kReserved; i < test::hdr::kCrc; ++i) reserved_zero &= f[i] == 0;
    CHECK(reserved_zero);

    uint64_t prev_end = test::hdr::kSize;
    for (int s = 0; s < 4; ++s) {
      const uint64_t off = test::section_offset(f, s), len = test::section_bytes(f, s);
      CHECK_EQ(off % 64, 0u);
      CHECK(off >= prev_end && off - prev_end < 64);  // the writer pads minimally
      prev_end = off + len;
    }
    CHECK_EQ(prev_end, uint64_t{f.size()});
    CHECK_EQ(test::section_bytes(f, test::hdr::kLevels), uint64_t{n});
    CHECK_EQ(test::section_bytes(f, test::hdr::kArena), uint64_t{n * index.layout().record_stride});
    CHECK_EQ(test::section_bytes(f, test::hdr::kUpperOffsets), uint64_t{n * 8});
    if (n == 0) continue;
    CHECK_EQ(test::get_field<uint32_t>(f, test::hdr::kEntry), index.entry_point());

    const GraphView g = index.view();
    CHECK(std::memcmp(f.data() + test::section_offset(f, test::hdr::kArena), g.base,
                      n * index.layout().record_stride) == 0);
    CHECK(std::memcmp(f.data() + test::section_offset(f, test::hdr::kLevels), g.levels, n) == 0);
    CHECK(std::memcmp(f.data() + test::section_offset(f, test::hdr::kUpperOffsets),
                      g.upper_offsets, n * 8) == 0);
    const LoadedIndex mapped = LoadedIndex::open(path);
    CHECK_EQ(reinterpret_cast<uintptr_t>(mapped.view().base) % 64, 0u);
    CHECK_EQ(mapped.file_size(), f.size());
  }
}

// Pins the exact bytes of a small deterministic file (scalar kernel, exact
// input data, so the same on every compiler). If this changes, files written
// by earlier builds may no longer load or may mean something else: any
// intentional change to the bytes written requires bumping format::kVersion
// (and the loader's accepted version), then updating this constant.
static void test_format_stability() {
  const TempDir dir;
  const HnswIndex index = test::build_exact(40, 6, 4, 16, 7, l2_squared_scalar);
  const std::string path = dir.file("stable.idx");
  index.save(path);
  const std::vector<uint8_t> f = test::read_bytes(path);
  uint64_t h = 1469598103934665603ull;  // FNV-1a 64
  for (uint8_t b : f) h = (h ^ b) * 1099511628211ull;
  constexpr uint64_t kVersion1Hash = 0x1475ed65e86045fbull;
  CHECK_EQ(h, kVersion1Hash);
  if (h != kVersion1Hash) {
    std::fprintf(stderr, "  file hash 0x%016" PRIx64 " (%zu bytes)\n", h, f.size());
  }
}

static void test_atomic_save() {
  const HnswIndex index = test::build_exact(50, 4, 4, 20, 3, l2_squared_scalar);
  const TempDir dir;

  CHECK_THROWS(index.save(dir.file("missing/sub/x.idx")), IndexFileError, "temporary file");
  CHECK(dir.entries().empty());

  // Target is a directory: the file is written, the rename fails, and the
  // temporary is removed.
  fs::create_directory(dir.file("target"));
  CHECK_THROWS(index.save(dir.file("target")), IndexFileError, "rename");
  CHECK(dir.entries() == std::vector<std::string>{"target"});
  fs::remove(dir.file("target"));

  if (::geteuid() != 0) {  // root ignores directory permissions
    fs::create_directory(dir.file("ro"));
    fs::permissions(dir.file("ro"), fs::perms::owner_read | fs::perms::owner_exec);
    CHECK_THROWS(index.save(dir.file("ro/x.idx")), IndexFileError, "Permission denied");
    fs::permissions(dir.file("ro"), fs::perms::owner_all);
    CHECK(fs::is_empty(dir.file("ro")));
    fs::remove(dir.file("ro"));
  }

  index.save(dir.file("ok.idx"));
  CHECK(dir.entries() == std::vector<std::string>{"ok.idx"});
}

// Replacing a file that is mapped leaves the mapping on the old contents.
static void test_overwrite_while_mapped() {
  const TempDir dir;
  const std::string path = dir.file("idx");
  const HnswIndex a = test::build_exact(200, 8, 6, 30, 1, best_l2_squared());
  const HnswIndex b = test::build_exact(300, 8, 6, 30, 2, best_l2_squared());
  a.save(path);
  const LoadedIndex old_map = LoadedIndex::open(path);
  b.save(path);
  check_same(a, old_map);
  check_same(b, LoadedIndex::open(path));
  CHECK(dir.entries() == std::vector<std::string>{"idx"});
}

static void test_load_twice_and_move() {
  const TempDir dir;
  const std::string path = dir.file("idx");
  const HnswIndex index = test::build_exact(500, 16, 8, 40, 5, best_l2_squared());
  index.save(path);
  LoadOptions read;
  read.mode = LoadMode::Read;
  LoadedIndex m1 = LoadedIndex::open(path);
  const LoadedIndex m2 = LoadedIndex::open(path);
  LoadedIndex r1 = LoadedIndex::open(path, read);
  check_same(index, m1);
  check_same(index, m2);
  check_same(index, r1);

  LoadedIndex moved(std::move(m1));
  CHECK_EQ(m1.size(), size_t{0});
  CHECK(m1.search(index.view().vector(0), 5, 10).ids.empty());
  check_same(index, moved);
  m1 = std::move(r1);  // into a moved-from object
  check_same(index, m1);
  moved = std::move(m1);  // releases the old mapping, takes the buffer
  CHECK(moved.mode() == LoadMode::Read);
  check_same(index, moved);
}

static void test_shallow_only_and_accessors() {
  const TempDir dir;
  const std::string path = dir.file("idx");
  const HnswIndex index = test::build_exact(120, 5, 4, 20, 9, best_l2_squared());
  index.save(path);
  LoadOptions shallow;
  shallow.validate_deep = false;
  const LoadedIndex loaded = LoadedIndex::open(path, shallow);
  check_same(index, loaded);

  CHECK_THROWS(loaded.node_level(120), std::out_of_range, "out of range");
  CHECK_THROWS(loaded.vector(120), std::out_of_range, "out of range");
  CHECK_THROWS(loaded.neighbors(0, index.node_level(0) + 1), std::out_of_range, "no level");
  CHECK_THROWS(loaded.neighbors(0, -1), std::out_of_range, "no level");

  const std::string empty_path = dir.file("empty");
  HnswIndex(3).save(empty_path);
  const LoadedIndex empty = LoadedIndex::open(empty_path);
  CHECK_EQ(empty.size(), size_t{0});
  CHECK_EQ(empty.max_level(), -1);
  CHECK_EQ(empty.file_size(), size_t{192});
  CHECK_THROWS(empty.entry_point(), std::logic_error, "empty");
  const float q[3] = {0, 0, 0};
  CHECK(empty.search(q, 3, 10).ids.empty());

  LoadOptions null_dist;
  null_dist.dist = nullptr;
  CHECK_THROWS(LoadedIndex::open(path, null_dist), std::invalid_argument, "null");
  CHECK_THROWS(LoadedIndex::open(dir.file("nope")), IndexFileError, "cannot open");
  CHECK_THROWS(LoadedIndex::open(dir.path()), IndexFileError, "not a regular file");
}

static void test_save_limits() {
  const TempDir dir;
  CHECK_THROWS(HnswIndex(format::kMaxDim + 1).save(dir.file("a")), IndexFileError, "dim");
  HnswParams p;
  p.M = format::kMaxM + 1;
  CHECK_THROWS(HnswIndex(1, p).save(dir.file("b")), IndexFileError, "M exceeds");
  CHECK(dir.entries().empty());
  HnswIndex(format::kMaxDim).save(dir.file("c"));
  p.M = format::kMaxM;
  HnswIndex(1, p).save(dir.file("d"));
  CHECK_EQ(LoadedIndex::open(dir.file("c")).dim(), format::kMaxDim);
  CHECK_EQ(LoadedIndex::open(dir.file("d")).M(), format::kMaxM);
}

int main() {
  test_crc_reference();
  test_round_trip();
  test_byte_layout();
  test_format_stability();
  test_atomic_save();
  test_overwrite_while_mapped();
  test_load_twice_and_move();
  test_shallow_only_and_accessors();
  test_save_limits();
  return test_finish();
}
