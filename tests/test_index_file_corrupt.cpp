// The loader treats files as untrusted input: truncated, bit-flipped and
// crafted files must throw IndexFileError (never crash), or load and then
// search safely; header-driven sizes must never cause large allocations.
// Run under ASan+UBSan to make "safely" meaningful.

#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "alloc_hook.h"
#include "hnsw/hnsw.h"
#include "hnsw/index_file.h"
#include "index_file_util.h"
#include "test_helper.h"

using namespace hnsw;
using test::hdr::kArena;
using test::hdr::kLevels;
using test::hdr::kUpperLinks;
using test::hdr::kUpperOffsets;

namespace {

constexpr LoadMode kModes[] = {LoadMode::Mmap, LoadMode::Read};

enum class Outcome { kThrew, kLoaded, kWrongException };

// Opens `path`; on success runs a few searches.
Outcome try_load(const std::string& path, LoadMode mode, std::string* message = nullptr) {
  LoadOptions opt;
  opt.mode = mode;
  try {
    const LoadedIndex index = LoadedIndex::open(path, opt);
    test::probe_searches(index);
    return Outcome::kLoaded;
  } catch (const IndexFileError& e) {
    if (message != nullptr) *message = e.what();
    return Outcome::kThrew;
  } catch (...) {
    return Outcome::kWrongException;
  }
}

// A small valid file with upper layers (M = 2 puts about half the nodes on
// layer 1 or above).
std::vector<uint8_t> small_file(const std::string& path) {
  const HnswIndex index = test::build_exact(30, 3, 2, 10, 17, l2_squared_scalar);
  index.save(path);
  return test::read_bytes(path);
}

// Parsed view of a valid file, for crafting data-section corruptions.
struct Parsed {
  explicit Parsed(const std::vector<uint8_t>& f)
      : n(test::get_field<uint64_t>(f, test::hdr::kCount)),
        stride(test::get_field<uint64_t>(f, test::hdr::kStride)),
        layout(Layout::make(test::get_field<uint32_t>(f, test::hdr::kDim),
                            test::get_field<uint32_t>(f, test::hdr::kM))),
        max_level(test::get_field<int32_t>(f, test::hdr::kMaxLevel)) {}
  size_t level(const std::vector<uint8_t>& f, size_t id) const {
    return f[test::section_offset(f, kLevels) + id];
  }
  size_t record(const std::vector<uint8_t>& f, size_t id) const {
    return test::section_offset(f, kArena) + id * stride;
  }
  size_t upper_offset_pos(const std::vector<uint8_t>& f, size_t id) const {
    return test::section_offset(f, kUpperOffsets) + id * 8;
  }
  // Byte position of the [count][ids] block of id on level >= 1.
  size_t upper_block(const std::vector<uint8_t>& f, size_t id, int level) const {
    const uint64_t off = test::get_field<uint64_t>(f, upper_offset_pos(f, id));
    return test::section_offset(f, kUpperLinks) + (off + (level - 1) * layout.upper_block) * 4;
  }
  size_t n, stride;
  Layout layout;
  int max_level;
};

}  // namespace

// (a) Every truncation, from 0 bytes to one byte short, throws.
static void test_truncation() {
  const test::TempFile src("trunc_src"), cut("trunc");
  const std::vector<uint8_t> f = small_file(src.path());
  size_t not_thrown = 0;
  for (size_t len = 0; len < f.size(); ++len) {
    test::write_bytes(cut.path(), f, len);
    for (LoadMode mode : kModes) not_thrown += try_load(cut.path(), mode) != Outcome::kThrew;
  }
  CHECK_EQ(not_thrown, size_t{0});
  test::write_bytes(cut.path(), f);
  for (LoadMode mode : kModes) CHECK(try_load(cut.path(), mode) == Outcome::kLoaded);
  // One byte too many is rejected as well.
  std::vector<uint8_t> longer = f;
  longer.push_back(0);
  test::write_bytes(cut.path(), longer);
  for (LoadMode mode : kModes) CHECK(try_load(cut.path(), mode) == Outcome::kThrew);
}

// (b) Every single-byte corruption either throws IndexFileError or loads and
// searches cleanly (deep validation is on). Header bytes are covered by the
// CRC, so corrupting any of them must throw.
static void test_byte_flips() {
  const test::TempFile src("flip_src"), bad("flip");
  const std::vector<uint8_t> f = small_file(src.path());
  const uint8_t patterns[] = {0xFF, 0x01, 0x80};
  size_t wrong_exception = 0, header_loaded = 0, loaded = 0, threw = 0;
  for (size_t i = 0; i < f.size(); ++i) {
    for (uint8_t x : patterns) {
      std::vector<uint8_t> g = f;
      g[i] ^= x;
      test::write_bytes(bad.path(), g);
      for (LoadMode mode : kModes) {
        const Outcome o = try_load(bad.path(), mode);
        wrong_exception += o == Outcome::kWrongException;
        loaded += o == Outcome::kLoaded;
        threw += o == Outcome::kThrew;
        if (i < test::hdr::kSize) header_loaded += o != Outcome::kThrew;
      }
    }
  }
  CHECK_EQ(wrong_exception, size_t{0});
  CHECK_EQ(header_loaded, size_t{0});
  CHECK(loaded > 0);  // e.g. flipped vector mantissas and padding bytes
  CHECK(threw > 0);
}

// (c) Crafted files: each must throw the named error quickly, without any
// allocation sized by a header field.
static void test_crafted() {
  const test::TempFile src("craft_src"), bad("craft");
  const std::vector<uint8_t> f = small_file(src.path());
  const Parsed p(f);
  constexpr size_t kAllocLimit = 64 * 1024;  // the file is ~2.5 KB
  CHECK(f.size() < kAllocLimit / 4);

  auto expect = [&](const std::vector<uint8_t>& g, const std::string& substr, const char* what) {
    test::write_bytes(bad.path(), g);
    for (LoadMode mode : kModes) {
      test::g_max_allocation = 0;
      std::string msg;
      const Outcome o = try_load(bad.path(), mode, &msg);
      if (o != Outcome::kThrew || msg.find(substr) == std::string::npos ||
          test::g_max_allocation >= kAllocLimit) {
        test::fail(__FILE__, __LINE__,
                   std::string(what) + ": expected '" + substr + "', got '" + msg + "'");
      }
      ++test::checks();
    }
  };
  // Mutates the header, fixes the CRC, and expects `substr`.
  auto header = [&](const std::function<void(std::vector<uint8_t>&)>& mutate,
                    const std::string& substr, const char* what) {
    std::vector<uint8_t> g = f;
    mutate(g);
    test::fix_crc(g);
    expect(g, substr, what);
  };
  // Mutates data sections only (the CRC covers the header alone).
  auto data = [&](const std::function<void(std::vector<uint8_t>&)>& mutate,
                  const std::string& substr, const char* what) {
    std::vector<uint8_t> g = f;
    mutate(g);
    expect(g, substr, what);
  };
  using V = std::vector<uint8_t>;
  using test::set_field;
  namespace h = test::hdr;
  auto sec_off = [](int s) { return h::kSections + 16 * s; };
  auto sec_len = [](int s) { return h::kSections + 16 * s + 8; };
  constexpr uint64_t kU64Max = std::numeric_limits<uint64_t>::max();
  constexpr uint32_t kU32Max = std::numeric_limits<uint32_t>::max();

  expect(V{}, "less than the 192-byte header", "zero-length file");
  expect(V(f.begin(), f.begin() + 191), "less than", "short header");
  { V g = f; g[0] ^= 1; expect(g, "bad magic", "magic"); }
  header([](V& g) { set_field<uint32_t>(g, h::kVersion, 2); }, "unsupported format version 2", "version");
  header([](V& g) { set_field<uint32_t>(g, h::kVersion, 0); }, "unsupported format version 0", "version 0");
  header([](V& g) { set_field<uint32_t>(g, h::kEndian, 0x04030201); }, "endianness", "big-endian marker");
  { V g = f; g[h::kCrc] ^= 1; expect(g, "CRC", "crc field"); }
  { V g = f; g[h::kDim] ^= 1; expect(g, "CRC", "field changed without CRC"); }
  header([](V& g) { g[h::kReserved + 5] = 1; }, "reserved header byte", "reserved");
  header([](V& g) { set_field<uint32_t>(g, h::kMetric, 1); }, "unknown metric id 1", "metric");
  header([](V& g) { set_field<uint32_t>(g, h::kDim, 0); }, "dim 0 outside", "dim 0");
  header([](V& g) { set_field<uint32_t>(g, h::kDim, kU32Max); }, "dim 4294967295 outside", "huge dim");
  header([](V& g) { set_field<uint32_t>(g, h::kDim, 65537); }, "dim 65537 outside", "dim over limit");
  header([](V& g) { set_field<uint32_t>(g, h::kM, 1); }, "M 1 outside", "M 1");
  header([](V& g) { set_field<uint32_t>(g, h::kM, kU32Max); }, "M 4294967295 outside", "huge M");
  header([](V& g) { set_field<uint64_t>(g, h::kCount, uint64_t{1} << 40); }, "exceeds uint32", "huge n");
  header([](V& g) { set_field<uint64_t>(g, h::kCount, kU64Max); }, "exceeds uint32", "n = 2^64-1");
  // Largest n the id range allows: passes the count check, fails on section lengths.
  header([](V& g) { set_field<uint64_t>(g, h::kCount, kU32Max); }, "has length", "n = 2^32-1");
  header([](V& g) { set_field<uint32_t>(g, h::kEntry, 30); }, "entry point 30 out of range", "entry");
  header([](V& g) { set_field<int32_t>(g, h::kMaxLevel, 256); }, "max level 256 outside", "max level");
  header([](V& g) { set_field<int32_t>(g, h::kMaxLevel, -1); }, "max level -1 outside", "max level -1");
  header([](V& g) { set_field<uint64_t>(g, h::kStride, 128); }, "record stride 128", "stride");
  header([&](V& g) { set_field<uint64_t>(g, h::kFileSize, f.size() + 64); }, "file size", "file size");
  header([&](V& g) { set_field<uint64_t>(g, h::kFileSize, kU64Max); }, "file size", "huge file size");
  header([&](V& g) { set_field<uint64_t>(g, sec_off(kLevels), 193); }, "levels section offset 193 is not 64-byte aligned", "misaligned");
  header([&](V& g) { set_field<uint64_t>(g, sec_off(kArena), test::section_offset(f, kArena) + 32); }, "not 64-byte aligned", "misaligned arena");
  header([&](V& g) { set_field<uint64_t>(g, sec_off(kLevels), 128); }, "overlaps the header", "section over header");
  header([&](V& g) { set_field<uint64_t>(g, sec_off(kArena), test::section_offset(f, kLevels)); }, "overlaps", "overlap");
  header([&](V& g) { set_field<uint64_t>(g, sec_off(kUpperLinks), (f.size() + 64) / 64 * 64); }, "extends past the end", "outside the file");
  header([&](V& g) { set_field<uint64_t>(g, sec_off(kUpperLinks), uint64_t{1} << 62); }, "extends past the end", "far outside");
  header([&](V& g) { set_field<uint64_t>(g, sec_len(kUpperLinks), kU64Max - 3); }, "overflows 64 bits", "offset + length overflows");
  header([&](V& g) { set_field<uint64_t>(g, sec_off(kUpperLinks), kU64Max - 63); }, "overflows 64 bits", "offset overflows");
  header([&](V& g) { set_field<uint64_t>(g, sec_len(kLevels), p.n + 1); }, "levels section has length", "levels length");
  header([&](V& g) { set_field<uint64_t>(g, sec_len(kArena), p.n * p.stride - 64); }, "layer-0 arena section has length", "arena length");
  header([&](V& g) { set_field<uint64_t>(g, sec_len(kUpperOffsets), 8); }, "upper offsets section has length", "offsets length");
  header([&](V& g) { set_field<uint64_t>(g, sec_len(kUpperLinks), test::section_bytes(f, kUpperLinks) - 2); }, "upper links section has length", "upper length % 4");
  header([&](V& g) { set_field<uint64_t>(g, sec_len(kUpperLinks), test::section_bytes(f, kUpperLinks) - 4); }, "node levels require", "upper length short");
  {  // a valid-looking empty index header that claims a max level
    V g(f.begin(), f.begin() + h::kSize);
    set_field<uint64_t>(g, h::kCount, 0);
    set_field<uint64_t>(g, h::kFileSize, h::kSize);
    for (int s = 0; s < 4; ++s) {
      set_field<uint64_t>(g, sec_off(s), h::kSize);
      set_field<uint64_t>(g, sec_len(s), 0);
    }
    set_field<uint32_t>(g, h::kEntry, 0);
    test::fix_crc(g);
    test::write_bytes(bad.path(), g);
    for (LoadMode mode : kModes) CHECK(try_load(bad.path(), mode) == Outcome::kThrew);
    set_field<int32_t>(g, h::kMaxLevel, -1);
    test::fix_crc(g);
    test::write_bytes(bad.path(), g);
    for (LoadMode mode : kModes) CHECK(try_load(bad.path(), mode) == Outcome::kLoaded);
  }

  // Deep validation: data sections.
  size_t level0 = p.n, upper = p.n, upper_low = p.n;
  for (size_t id = 0; id < p.n; ++id) {
    if (p.level(f, id) == 0 && level0 == p.n) level0 = id;
    if (p.level(f, id) >= 1 && upper == p.n &&
        test::get_field<uint32_t>(f, p.upper_block(f, id, 1)) > 0) {
      upper = id;
    }
  }
  for (size_t id = 0; id < p.n; ++id) {
    if (p.level(f, id) < 1 && id != upper) upper_low = id;
  }
  CHECK(level0 < p.n && upper < p.n && upper_low < p.n && p.max_level >= 1);
  if (level0 >= p.n || upper >= p.n || upper_low >= p.n) return;
  const size_t r0 = p.record(f, level0);
  const size_t links0 = r0 + p.layout.links_offset;
  const uint32_t cap0 = static_cast<uint32_t>(p.layout.max_links0);

  data([&](V& g) { set_field<uint32_t>(g, r0, cap0 + 1); }, "links, cap", "layer-0 count > cap");
  data([&](V& g) { set_field<uint32_t>(g, r0, kU32Max); }, "links, cap", "huge layer-0 count");
  data([&](V& g) { set_field<uint32_t>(g, r0, 1); set_field<uint32_t>(g, links0, 30); }, "links invalid id 30", "id >= n");
  data([&](V& g) { set_field<uint32_t>(g, r0, 1); set_field<uint32_t>(g, links0, kU32Max); }, "invalid id", "id = UINT32_MAX");
  data([&](V& g) { set_field<uint32_t>(g, r0, 1); set_field<uint32_t>(g, links0, static_cast<uint32_t>(level0)); }, "self-loop", "self-loop");
  const size_t ub = p.upper_block(f, upper, 1);
  data([&](V& g) { set_field<uint32_t>(g, ub, static_cast<uint32_t>(p.layout.M) + 1); }, "links, cap", "upper count > cap");
  data([&](V& g) { set_field<uint32_t>(g, ub + 4, static_cast<uint32_t>(upper_low)); }, "whose level is 0", "link to a node without the layer");
  data([&](V& g) { g[test::section_offset(f, kLevels) + level0] = static_cast<uint8_t>(p.max_level + 1); }, "above the max level", "level > max");
  data([&](V& g) { g[test::section_offset(f, kLevels) + level0] = 1; }, "upper offset", "level raised without blocks");
  data([&](V& g) { set_field<uint64_t>(g, p.upper_offset_pos(f, level0), 1); }, "level 0 but a nonzero upper offset", "stray offset");
  data([&](V& g) { set_field<uint64_t>(g, p.upper_offset_pos(f, upper), uint64_t{1} << 60); }, "upper offset", "offset out of range");
  {
    const uint32_t entry = test::get_field<uint32_t>(f, test::hdr::kEntry);
    data([&](V& g) { g[test::section_offset(f, kLevels) + entry] = 0; }, "entry point", "entry not on max level");
  }
  const size_t vec = r0 + p.layout.vector_offset;
  data([&](V& g) { set_field<float>(g, vec + 4, std::numeric_limits<float>::quiet_NaN()); }, "vector element 1 is not finite", "NaN");
  data([&](V& g) { set_field<float>(g, vec, -std::numeric_limits<float>::infinity()); }, "vector element 0 is not finite", "-inf");
  data([&](V& g) { set_field<uint32_t>(g, vec + 8, 0x7F800001u); }, "not finite", "signaling NaN");

  // The same corruptions pass with deep validation off: it only checks the
  // header and section bounds (which is why such files must be trusted).
  V g = f;
  set_field<uint32_t>(g, r0, kU32Max);
  test::write_bytes(bad.path(), g);
  LoadOptions shallow;
  shallow.validate_deep = false;
  bool loads = true;
  try {
    LoadedIndex::open(bad.path(), shallow);
  } catch (...) {
    loads = false;
  }
  CHECK(loads);
}

int main() {
  test_truncation();
  test_byte_flips();
  test_crafted();
  return test_finish();
}
