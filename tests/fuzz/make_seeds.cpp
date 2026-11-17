// Writes the seed corpora for the fuzz targets (run by gen_seeds.sh).
//
// Usage: make_seeds <loader_dir> <fvecs_dir>
//
// Loader seeds are small valid index files written by HnswIndex::save from
// deterministic data (n = 0, 1, 2 and up to a few dozen; several dim and M;
// with and without upper layers), plus a few invalid variants: truncated, one
// byte too long, and a wrong header CRC. fvecs seeds are small valid and
// truncated vector files.

#include <cstdio>
#include <string>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw/io.h"
#include "index_file_util.h"

namespace {

void write(const std::string& dir, const std::string& name, const std::vector<uint8_t>& bytes) {
  test::write_bytes(dir + "/" + name, bytes);
}

std::vector<uint8_t> prefix(const std::vector<uint8_t>& b, size_t n) {
  return {b.begin(), b.begin() + static_cast<std::ptrdiff_t>(n)};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <loader_dir> <fvecs_dir>\n", argv[0]);
    return 2;
  }
  const std::string loader = argv[1], fvecs = argv[2];
  const std::string tmp = loader + "/.make_seeds.tmp";

  struct Config {
    size_t n, dim, M;
  };
  const Config configs[] = {{0, 3, 2},  {1, 1, 2},  {2, 4, 4},   {3, 2, 2},
                            {8, 5, 16}, {24, 4, 2}, {30, 3, 2},  {40, 16, 4}};
  bool flat = false, layered = false;
  std::vector<uint8_t> largest_layered;
  for (const Config& c : configs) {
    const hnsw::HnswIndex index =
        test::build_exact(c.n, c.dim, c.M, 16, 11 + c.n, hnsw::l2_squared_scalar);
    index.save(tmp);
    const std::vector<uint8_t> bytes = test::read_bytes(tmp);
    if (c.n >= 2 && index.max_level() == 0) flat = true;
    if (index.max_level() > 0) {
      layered = true;
      if (bytes.size() > largest_layered.size()) largest_layered = bytes;
    }
    write(loader, "valid_n" + std::to_string(c.n) + "_d" + std::to_string(c.dim) + "_M" +
                      std::to_string(c.M) + "_L" + std::to_string(index.max_level()),
          bytes);
  }
  std::remove(tmp.c_str());
  if (!flat || !layered) {
    std::fprintf(stderr, "make_seeds: configs no longer cover flat and layered graphs\n");
    return 1;
  }

  const std::vector<uint8_t>& f = largest_layered;
  write(loader, "trunc_header", prefix(f, 191));
  write(loader, "trunc_sections", prefix(f, 256));
  write(loader, "trunc_last_byte", prefix(f, f.size() - 1));
  std::vector<uint8_t> longer = f;
  longer.push_back(0);
  write(loader, "extra_byte", longer);
  std::vector<uint8_t> bad_crc = f;
  bad_crc[test::hdr::kCrc] ^= 1;
  write(loader, "bad_crc", bad_crc);

  const std::vector<float> v = test::exact_data(6, 3, 5);
  hnsw::write_fvecs(fvecs + "/valid_f_d3_n6", v, 3);
  hnsw::write_fvecs(fvecs + "/valid_f_d1_n1", {0.5f}, 1);
  hnsw::write_ivecs(fvecs + "/valid_i_d4_n2", {0, 1, 2, 3, -1, 7, 1 << 20, 9}, 4);
  const std::vector<uint8_t> fv = test::read_bytes(fvecs + "/valid_f_d3_n6");
  write(fvecs, "trunc_record", prefix(fv, fv.size() - 2));
  write(fvecs, "trunc_dim", prefix(fv, 2));
  std::vector<uint8_t> mixed = fv;
  mixed[16] = 4;  // record 1 declares dim 4
  write(fvecs, "mixed_dims", mixed);
  return 0;
}
