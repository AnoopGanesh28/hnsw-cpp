// Fuzz target for the fvecs/ivecs readers. The input is the file content,
// served from an in-memory file since the readers take paths. A successful
// read must describe exactly the input: count records of (dim, dim values).
//
// Built as a libFuzzer target (HNSW_BUILD_FUZZ) and, with replay_main.cpp,
// as the replay_fvecs regression test.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "hnsw/io.h"
#include "memfd.h"

namespace {

template <typename Set>
void check(const Set& s, const uint8_t* data, size_t size, const char* kind) {
  bool ok = s.dim > 0 && s.count > 0 && s.data.size() == s.count * s.dim &&
            size == s.count * (4 + 4 * s.dim);
  for (size_t i = 0; ok && i < s.count; ++i) {
    const uint8_t* rec = data + i * (4 + 4 * s.dim);
    int32_t d;
    std::memcpy(&d, rec, 4);
    ok = static_cast<size_t>(d) == s.dim &&
         std::memcmp(rec + 4, s.data.data() + i * s.dim, 4 * s.dim) == 0;
  }
  if (!ok) {
    std::fprintf(stderr, "fuzz_fvecs: %s result does not match the input\n", kind);
    std::abort();
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  const fuzz::MemFile file(data, size);
  try {
    check(hnsw::read_fvecs(file.path()), data, size, "fvecs");
  } catch (const std::runtime_error&) {
  }
  try {
    check(hnsw::read_ivecs(file.path()), data, size, "ivecs");
  } catch (const std::runtime_error&) {
  }
  return 0;
}
