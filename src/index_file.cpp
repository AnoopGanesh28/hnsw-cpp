#include "hnsw/index_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <limits>
#include <new>
#include <system_error>
#include <utility>

#include "hnsw/hnsw.h"

namespace hnsw {
namespace {

static_assert(sizeof(size_t) == 8 && sizeof(float) == 4, "requires a 64-bit host");
constexpr bool kLittleEndianHost = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;

using format::kHeaderSize;
using format::kSectionAlign;

constexpr char kMagic[8] = {'H', 'N', 'S', 'W', 'I', 'D', 'X', '\0'};
constexpr uint32_t kEndianMarker = 0x01020304;
constexpr uint32_t kMetricL2Squared = 0;
constexpr int kMaxLevel = std::numeric_limits<uint8_t>::max();  // levels are stored as uint8

// Header field byte offsets; see docs/FORMAT.md.
constexpr size_t kOffMagic = 0;
constexpr size_t kOffVersion = 8;
constexpr size_t kOffEndian = 12;
constexpr size_t kOffMetric = 16;
constexpr size_t kOffDim = 20;
constexpr size_t kOffM = 24;
constexpr size_t kOffEfConstruction = 28;
constexpr size_t kOffCount = 32;
constexpr size_t kOffEntry = 40;
constexpr size_t kOffMaxLevel = 44;
constexpr size_t kOffStride = 48;
constexpr size_t kOffFileSize = 56;
constexpr size_t kOffSections = 64;  // 4 x (u64 offset, u64 length)
constexpr size_t kOffReserved = 128;
constexpr size_t kOffCrc = 188;
static_assert(kOffCrc + sizeof(uint32_t) == kHeaderSize, "CRC is the last header field");
static_assert(kHeaderSize % kSectionAlign == 0, "header is padded to the section alignment");

enum Section { kLevels, kArena, kUpperOffsets, kUpperLinks, kNumSections };
constexpr const char* kSectionNames[kNumSections] = {"levels", "layer-0 arena", "upper offsets",
                                                     "upper links"};

// CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320, init and final xor
// 0xFFFFFFFF), the zlib crc32.
constexpr std::array<uint32_t, 256> make_crc_table() {
  std::array<uint32_t, 256> t{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int b = 0; b < 8; ++b) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    t[i] = c;
  }
  return t;
}
constexpr std::array<uint32_t, 256> kCrcTable = make_crc_table();

uint32_t crc32(const uint8_t* p, size_t n) {
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) c = kCrcTable[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

// Little-endian field access (the host is little-endian).
template <typename T>
T get(const uint8_t* p, size_t off) {
  T v;
  std::memcpy(&v, p + off, sizeof(v));
  return v;
}
template <typename T>
void put(uint8_t* p, size_t off, T v) {
  std::memcpy(p + off, &v, sizeof(v));
}

struct Range {
  uint64_t offset = 0;
  uint64_t bytes = 0;
};

struct Header {
  uint32_t dim = 0;
  uint32_t M = 0;
  uint32_t ef_construction = 0;
  uint64_t count = 0;
  uint32_t entry = 0;
  int32_t max_level = -1;
  uint64_t stride = 0;
  uint64_t file_size = 0;
  Range sections[kNumSections];
};

// Formats errors as "hnsw index '<path>': <message>" and does overflow-checked
// arithmetic on untrusted values.
class Errors {
 public:
  explicit Errors(const std::string& path) : who_("hnsw index '" + path + "'") {}

  [[noreturn]] void fail(const std::string& msg) const { throw IndexFileError(who_ + ": " + msg); }
  [[noreturn]] void fail_errno(const std::string& msg, int err) const {
    fail(msg + ": " + std::system_category().message(err));
  }

  uint64_t mul(uint64_t a, uint64_t b, const char* what) const {
    uint64_t r;
    if (__builtin_mul_overflow(a, b, &r)) fail(std::string(what) + " overflows 64 bits");
    return r;
  }
  uint64_t add(uint64_t a, uint64_t b, const char* what) const {
    uint64_t r;
    if (__builtin_add_overflow(a, b, &r)) fail(std::string(what) + " overflows 64 bits");
    return r;
  }

 private:
  std::string who_;
};

class Fd {
 public:
  explicit Fd(int fd) : fd_(fd) {}
  ~Fd() {
    if (fd_ >= 0) ::close(fd_);
  }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  int get() const { return fd_; }
  // Closes now, reporting the error (close can surface deferred write errors).
  int close() { return ::close(std::exchange(fd_, -1)); }

 private:
  int fd_;
};

void write_all(int fd, const void* data, size_t n, const Errors& err) {
  const auto* p = static_cast<const uint8_t*>(data);
  while (n > 0) {
    const ssize_t w = ::write(fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      err.fail_errno("write failed", errno);
    }
    p += w;
    n -= static_cast<size_t>(w);
  }
}

void read_all(int fd, uint8_t* p, size_t n, const Errors& err) {
  size_t done = 0;
  while (done < n) {
    const ssize_t r = ::pread(fd, p + done, n - done, static_cast<off_t>(done));
    if (r < 0) {
      if (errno == EINTR) continue;
      err.fail_errno("read failed", errno);
    }
    if (r == 0) err.fail("file shrank while being read");
    done += static_cast<size_t>(r);
  }
}

uint64_t round_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

void encode_header(const Header& h, uint8_t* out) {
  std::memset(out, 0, kHeaderSize);
  std::memcpy(out + kOffMagic, kMagic, sizeof(kMagic));
  put<uint32_t>(out, kOffVersion, format::kVersion);
  put<uint32_t>(out, kOffEndian, kEndianMarker);
  put<uint32_t>(out, kOffMetric, kMetricL2Squared);
  put<uint32_t>(out, kOffDim, h.dim);
  put<uint32_t>(out, kOffM, h.M);
  put<uint32_t>(out, kOffEfConstruction, h.ef_construction);
  put<uint64_t>(out, kOffCount, h.count);
  put<uint32_t>(out, kOffEntry, h.entry);
  put<int32_t>(out, kOffMaxLevel, h.max_level);
  put<uint64_t>(out, kOffStride, h.stride);
  put<uint64_t>(out, kOffFileSize, h.file_size);
  for (size_t s = 0; s < kNumSections; ++s) {
    put<uint64_t>(out, kOffSections + 16 * s, h.sections[s].offset);
    put<uint64_t>(out, kOffSections + 16 * s + 8, h.sections[s].bytes);
  }
  put<uint32_t>(out, kOffCrc, crc32(out, kOffCrc));
}

// Shallow validation: everything that can be checked from the header and the
// real file size alone, in O(1). Afterwards every section lies inside the
// file, is aligned, and has exactly the length the header's counts imply.
Header decode_header(const uint8_t* p, uint64_t real_size, const Errors& err) {
  if (std::memcmp(p + kOffMagic, kMagic, sizeof(kMagic)) != 0) err.fail("bad magic");
  const uint32_t version = get<uint32_t>(p, kOffVersion);
  if (version != format::kVersion) {
    err.fail("unsupported format version " + std::to_string(version) + " (expected " +
             std::to_string(format::kVersion) + ")");
  }
  if (get<uint32_t>(p, kOffEndian) != kEndianMarker) {
    err.fail("endianness marker mismatch (file is not little-endian)");
  }
  if (get<uint32_t>(p, kOffCrc) != crc32(p, kOffCrc)) err.fail("header CRC mismatch");
  for (size_t i = kOffReserved; i < kOffCrc; ++i) {
    if (p[i] != 0) err.fail("reserved header byte " + std::to_string(i) + " is not zero");
  }
  const uint32_t metric = get<uint32_t>(p, kOffMetric);
  if (metric != kMetricL2Squared) err.fail("unknown metric id " + std::to_string(metric));

  Header h;
  h.dim = get<uint32_t>(p, kOffDim);
  h.M = get<uint32_t>(p, kOffM);
  h.ef_construction = get<uint32_t>(p, kOffEfConstruction);
  h.count = get<uint64_t>(p, kOffCount);
  h.entry = get<uint32_t>(p, kOffEntry);
  h.max_level = get<int32_t>(p, kOffMaxLevel);
  h.stride = get<uint64_t>(p, kOffStride);
  h.file_size = get<uint64_t>(p, kOffFileSize);
  for (size_t s = 0; s < kNumSections; ++s) {
    h.sections[s].offset = get<uint64_t>(p, kOffSections + 16 * s);
    h.sections[s].bytes = get<uint64_t>(p, kOffSections + 16 * s + 8);
  }

  if (h.dim < 1 || h.dim > format::kMaxDim) {
    err.fail("dim " + std::to_string(h.dim) + " outside [1, " + std::to_string(format::kMaxDim) + "]");
  }
  if (h.M < 2 || h.M > format::kMaxM) {
    err.fail("M " + std::to_string(h.M) + " outside [2, " + std::to_string(format::kMaxM) + "]");
  }
  if (h.count > std::numeric_limits<uint32_t>::max()) {
    err.fail("element count " + std::to_string(h.count) + " exceeds uint32 ids");
  }
  if (h.count == 0) {
    if (h.max_level != -1 || h.entry != 0) err.fail("empty index must have max level -1 and entry 0");
  } else {
    if (h.max_level < 0 || h.max_level > kMaxLevel) {
      err.fail("max level " + std::to_string(h.max_level) + " outside [0, " +
               std::to_string(kMaxLevel) + "]");
    }
    if (h.entry >= h.count) err.fail("entry point " + std::to_string(h.entry) + " out of range");
  }
  const uint64_t stride = Layout::make(h.dim, h.M).record_stride;
  if (h.stride != stride) {
    err.fail("record stride " + std::to_string(h.stride) + " does not match " +
             std::to_string(stride) + " computed from dim and M");
  }
  if (h.file_size != real_size) {
    err.fail("header records file size " + std::to_string(h.file_size) + " but the file has " +
             std::to_string(real_size) + " bytes");
  }

  const uint64_t expected[kNumSections] = {
      h.count, err.mul(h.count, h.stride, "layer-0 arena size"),
      err.mul(h.count, sizeof(uint64_t), "upper offset table size"), 0};
  uint64_t prev_end = kHeaderSize;
  for (size_t s = 0; s < kNumSections; ++s) {
    const Range& r = h.sections[s];
    const std::string name = std::string(kSectionNames[s]) + " section";
    if (r.offset % kSectionAlign != 0) {
      err.fail(name + " offset " + std::to_string(r.offset) + " is not " +
               std::to_string(kSectionAlign) + "-byte aligned");
    }
    if (r.offset < prev_end) err.fail(name + " overlaps the header or the previous section");
    const uint64_t end = err.add(r.offset, r.bytes, "section end");
    if (end > real_size) err.fail(name + " extends past the end of the file");
    if (s == kUpperLinks ? r.bytes % sizeof(uint32_t) != 0 : r.bytes != expected[s]) {
      err.fail(name + " has length " + std::to_string(r.bytes) + ", which does not match the header");
    }
    prev_end = end;
  }
  return h;
}

GraphView make_view(const uint8_t* file, const Header& h) {
  GraphView g;
  g.layout = Layout::make(h.dim, h.M);
  g.count = h.count;
  g.base = file + h.sections[kArena].offset;
  g.levels = file + h.sections[kLevels].offset;
  g.upper_offsets = reinterpret_cast<const uint64_t*>(file + h.sections[kUpperOffsets].offset);
  g.upper_links = reinterpret_cast<const uint32_t*>(file + h.sections[kUpperLinks].offset);
  g.entry_point = h.entry;
  g.max_level = h.max_level;
  return g;
}

// Deep validation, O(file size), no allocation. Establishes everything search
// relies on: the entry point is on the top layer; each node's upper blocks
// lie inside the upper table (they are dense, in id order); every link count
// is within its cap; every link points to another existing node that has the
// linked layer; every vector element is finite.
void validate_deep(const GraphView& g, const Header& h, const Errors& err) {
  const size_t n = g.count;
  const std::string node = "node ";
  for (size_t i = 0; i < n; ++i) {
    if (g.levels[i] > g.max_level) {
      err.fail(node + std::to_string(i) + " has level " + std::to_string(g.levels[i]) +
               " above the max level " + std::to_string(g.max_level));
    }
  }
  if (n > 0 && g.levels[g.entry_point] != g.max_level) {
    err.fail("entry point " + std::to_string(g.entry_point) + " is not on the max level");
  }

  uint64_t next = 0;  // < 2^32 nodes * 255 levels * 1025 words: no overflow
  for (size_t i = 0; i < n; ++i) {
    const uint64_t off = g.upper_offsets[i];
    if (g.levels[i] == 0) {
      if (off != 0) err.fail(node + std::to_string(i) + " has level 0 but a nonzero upper offset");
      continue;
    }
    if (off != next) {
      err.fail(node + std::to_string(i) + " upper offset " + std::to_string(off) + ", expected " +
               std::to_string(next));
    }
    next += uint64_t{g.levels[i]} * g.layout.upper_block;
  }
  if (next * sizeof(uint32_t) != h.sections[kUpperLinks].bytes) {
    err.fail("upper links section has " + std::to_string(h.sections[kUpperLinks].bytes) +
             " bytes, node levels require " + std::to_string(next * sizeof(uint32_t)));
  }

  for (uint32_t id = 0; id < n; ++id) {
    for (int l = 0; l <= g.levels[id]; ++l) {
      const NeighborView nb = g.neighbors(id, l);
      if (nb.count > g.layout.max_links(l)) {
        err.fail(node + std::to_string(id) + " level " + std::to_string(l) + " has " +
                 std::to_string(nb.count) + " links, cap " + std::to_string(g.layout.max_links(l)));
      }
      for (uint32_t x : nb) {
        if (x >= n) {
          err.fail(node + std::to_string(id) + " level " + std::to_string(l) + " links invalid id " +
                   std::to_string(x));
        }
        if (x == id) err.fail(node + std::to_string(id) + " level " + std::to_string(l) + " has a self-loop");
        if (g.levels[x] < l) {
          err.fail(node + std::to_string(id) + " level " + std::to_string(l) + " links node " +
                   std::to_string(x) + " whose level is " + std::to_string(g.levels[x]));
        }
      }
    }
    const uint8_t* v = g.record(id) + g.layout.vector_offset;
    for (size_t d = 0; d < g.layout.dim; ++d) {
      const uint32_t bits = get<uint32_t>(v, d * sizeof(float));
      if ((bits & 0x7F800000u) == 0x7F800000u) {
        err.fail(node + std::to_string(id) + " vector element " + std::to_string(d) +
                 " is not finite");
      }
    }
  }
}

}  // namespace

void HnswIndex::save(const std::string& path) const {
  const Errors err(path);
  if (dim_ > format::kMaxDim) err.fail("dim exceeds the format limit");
  if (params_.M > format::kMaxM) err.fail("M exceeds the format limit");
  if (params_.ef_construction > std::numeric_limits<uint32_t>::max()) {
    err.fail("ef_construction does not fit in 32 bits");
  }
  const size_t n = size();

  Header h;
  h.dim = static_cast<uint32_t>(dim_);
  h.M = static_cast<uint32_t>(params_.M);
  h.ef_construction = static_cast<uint32_t>(params_.ef_construction);
  h.count = n;
  h.entry = n == 0 ? 0 : entry_;
  h.max_level = max_level_;
  h.stride = layout_.record_stride;
  const uint64_t lengths[kNumSections] = {n, n * layout_.record_stride, n * sizeof(uint64_t),
                                          upper_links_.size() * sizeof(uint32_t)};
  const void* data[kNumSections] = {levels_.data(), arena_.data(), upper_offsets_.data(),
                                    upper_links_.data()};
  uint64_t end = kHeaderSize;
  for (size_t s = 0; s < kNumSections; ++s) {
    h.sections[s] = {round_up(end, kSectionAlign), lengths[s]};
    end = h.sections[s].offset + lengths[s];
  }
  h.file_size = end;
  uint8_t header[kHeaderSize];
  encode_header(h, header);

  // Unique name next to the target; O_EXCL never reuses an existing file.
  static std::atomic<unsigned> counter{0};
  std::string tmp;
  int raw = -1;
  for (int attempt = 0; raw < 0; ++attempt) {
    tmp = path + ".tmp-" + std::to_string(::getpid()) + "-" + std::to_string(counter++);
    raw = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
    if (raw < 0 && (errno != EEXIST || attempt >= 100)) {
      err.fail_errno("cannot create temporary file '" + tmp + "'", errno);
    }
  }
  Fd fd(raw);
  struct TempGuard {
    const std::string& path;
    bool keep = false;
    ~TempGuard() {
      if (!keep) ::unlink(path.c_str());
    }
  } guard{tmp};

  static constexpr uint8_t kZeros[kSectionAlign] = {};
  write_all(fd.get(), header, kHeaderSize, err);
  uint64_t pos = kHeaderSize;
  for (size_t s = 0; s < kNumSections; ++s) {
    write_all(fd.get(), kZeros, h.sections[s].offset - pos, err);
    if (lengths[s] > 0) write_all(fd.get(), data[s], lengths[s], err);
    pos = h.sections[s].offset + lengths[s];
  }
  if (::fsync(fd.get()) != 0) err.fail_errno("fsync failed", errno);
  if (fd.close() != 0) err.fail_errno("close failed", errno);
  if (::rename(tmp.c_str(), path.c_str()) != 0) err.fail_errno("cannot rename temporary file into place", errno);
  guard.keep = true;

  std::string dir = std::filesystem::path(path).parent_path().string();
  if (dir.empty()) dir = ".";
  Fd dfd(::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (dfd.get() < 0 || ::fsync(dfd.get()) != 0) {
    err.fail_errno("file written, but syncing its directory failed", errno);
  }
}

void LoadedIndex::AlignedFree::operator()(uint8_t* p) const {
  ::operator delete(p, std::align_val_t{kSectionAlign});
}

LoadedIndex LoadedIndex::parse(const uint8_t* data, size_t size, const LoadOptions& options,
                               const std::string& name) {
  if (options.dist == nullptr) {
    throw std::invalid_argument("LoadedIndex: distance function must not be null");
  }
  const Errors err(name);
  if (!kLittleEndianHost) err.fail("big-endian hosts are not supported");
  if (size < kHeaderSize) {
    err.fail("file has " + std::to_string(size) + " bytes, less than the " +
             std::to_string(kHeaderSize) + "-byte header");
  }
  if (reinterpret_cast<uintptr_t>(data) % kSectionAlign != 0) {
    err.fail("data is not " + std::to_string(kSectionAlign) + "-byte aligned");
  }

  LoadedIndex index;
  index.dist_ = options.dist;
  index.mode_ = LoadMode::Borrowed;
  index.bytes_ = size;
  const Header h = decode_header(data, size, err);
  index.view_ = make_view(data, h);
  index.ef_construction_ = h.ef_construction;
  if (options.validate_deep) validate_deep(index.view_, h, err);
  return index;
}

LoadedIndex LoadedIndex::from_memory(const void* data, size_t size, const LoadOptions& options) {
  return parse(static_cast<const uint8_t*>(data), size, options, "<memory>");
}

LoadedIndex LoadedIndex::open(const std::string& path, const LoadOptions& options) {
  if (options.dist == nullptr) {
    throw std::invalid_argument("LoadedIndex::open: distance function must not be null");
  }
  if (options.mode == LoadMode::Borrowed) {
    throw std::invalid_argument("LoadedIndex::open: mode must be Mmap or Read");
  }
  const Errors err(path);

  Fd fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
  if (fd.get() < 0) err.fail_errno("cannot open", errno);
  struct stat st;
  if (::fstat(fd.get(), &st) != 0) err.fail_errno("cannot stat", errno);
  if (!S_ISREG(st.st_mode)) err.fail("not a regular file");
  const uint64_t size = static_cast<uint64_t>(st.st_size);
  if (size < kHeaderSize) {  // also keeps a zero-length mmap from being attempted
    err.fail("file has " + std::to_string(size) + " bytes, less than the " +
             std::to_string(kHeaderSize) + "-byte header");
  }

  // `owner` holds the mapping or buffer, releasing it if parse() throws.
  LoadedIndex owner;
  owner.bytes_ = size;
  const uint8_t* file = nullptr;
  if (options.mode == LoadMode::Mmap) {
    void* m = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd.get(), 0);
    if (m == MAP_FAILED) err.fail_errno("mmap failed", errno);
    owner.map_ = m;
    file = static_cast<const uint8_t*>(m);
  } else {
    try {
      owner.buffer_.reset(
          static_cast<uint8_t*>(::operator new(size, std::align_val_t{kSectionAlign})));
    } catch (const std::bad_alloc&) {
      err.fail("cannot allocate " + std::to_string(size) + " bytes");
    }
    read_all(fd.get(), owner.buffer_.get(), size, err);
    file = owner.buffer_.get();
  }

  LoadedIndex index = parse(file, size, options, path);
  index.mode_ = options.mode;
  index.map_ = std::exchange(owner.map_, nullptr);
  index.buffer_ = std::move(owner.buffer_);
  return index;
}

LoadedIndex::LoadedIndex(LoadedIndex&& other) noexcept
    : view_(std::exchange(other.view_, GraphView{})),
      dist_(other.dist_),
      ef_construction_(other.ef_construction_),
      mode_(other.mode_),
      map_(std::exchange(other.map_, nullptr)),
      buffer_(std::move(other.buffer_)),
      bytes_(std::exchange(other.bytes_, 0)) {}

LoadedIndex& LoadedIndex::operator=(LoadedIndex&& other) noexcept {
  if (this != &other) {
    release();
    view_ = std::exchange(other.view_, GraphView{});
    dist_ = other.dist_;
    ef_construction_ = other.ef_construction_;
    mode_ = other.mode_;
    map_ = std::exchange(other.map_, nullptr);
    buffer_ = std::move(other.buffer_);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

LoadedIndex::~LoadedIndex() { release(); }

void LoadedIndex::release() noexcept {
  if (map_ != nullptr) ::munmap(map_, bytes_);
  map_ = nullptr;
  buffer_.reset();
  view_ = GraphView{};
  bytes_ = 0;
}

uint32_t LoadedIndex::entry_point() const {
  if (size() == 0) throw std::logic_error("LoadedIndex: index is empty");
  return view_.entry_point;
}

int LoadedIndex::node_level(uint32_t id) const {
  if (id >= size()) {
    throw std::out_of_range("LoadedIndex: id " + std::to_string(id) + " out of range");
  }
  return view_.levels[id];
}

NeighborView LoadedIndex::neighbors(uint32_t id, int level) const {
  if (level < 0 || level > node_level(id)) {
    throw std::out_of_range("LoadedIndex: node " + std::to_string(id) + " has no level " +
                            std::to_string(level));
  }
  return view_.neighbors(id, level);
}

const float* LoadedIndex::vector(uint32_t id) const {
  if (id >= size()) {
    throw std::out_of_range("LoadedIndex: id " + std::to_string(id) + " out of range");
  }
  return view_.vector(id);
}

KnnResult LoadedIndex::search(const float* query, size_t k, size_t ef) const {
  thread_local SearchScratch scratch;
  return search(query, k, ef, scratch);
}

KnnResult LoadedIndex::search(const float* query, size_t k, size_t ef,
                              SearchScratch& scratch) const {
  return search_graph(view_, dist_, query, k, ef, scratch);
}

size_t LoadedIndex::search_into(const float* query, size_t k, size_t ef, SearchScratch& scratch,
                                uint32_t* out_ids, float* out_dists) const {
  return search_graph_into(view_, dist_, query, k, ef, scratch, out_ids, out_dists);
}

void LoadedIndex::search_batch(const float* queries, size_t n_queries, size_t k, size_t ef,
                               uint32_t* out_ids, float* out_dists, ThreadPool& pool,
                               size_t chunk) const {
  search_graph_batch(view_, dist_, queries, n_queries, k, ef, out_ids, out_dists, pool, chunk);
}

}  // namespace hnsw
