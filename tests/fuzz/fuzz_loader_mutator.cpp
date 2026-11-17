// Custom libFuzzer mutator for fuzz_loader (linked into the libFuzzer binary
// only). The header CRC covers bytes [0, 188), so a random mutation of the
// header almost never passes the loader's CRC check and the checks behind it
// would stay unexplored. This mutates with libFuzzer's own mutators, then
// rewrites the CRC field to match. The loader's CRC check is unchanged; seeds
// with a wrong CRC (tests/fuzz/seeds/loader) keep that branch covered.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "index_file_util.h"

extern "C" size_t LLVMFuzzerMutate(uint8_t* data, size_t size, size_t max_size);

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t* data, size_t size, size_t max_size,
                                          unsigned int /*seed*/) {
  size = LLVMFuzzerMutate(data, size, max_size);
  if (size >= test::hdr::kSize) {
    const uint32_t crc = test::crc32_bitwise(data, test::hdr::kCrc);
    std::memcpy(data + test::hdr::kCrc, &crc, sizeof(crc));
  }
  return size;
}
