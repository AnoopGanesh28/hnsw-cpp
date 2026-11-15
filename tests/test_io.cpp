#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include "hnsw/io.h"
#include "test_helper.h"

using namespace hnsw;

static void write_raw(const std::string& path, const std::vector<int32_t>& words,
                      size_t extra_bytes = 0) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(words.data()),
            static_cast<std::streamsize>(words.size() * sizeof(int32_t)));
  for (size_t i = 0; i < extra_bytes; ++i) out.put('\0');
}

static void test_fvecs_round_trip() {
  test::TempFile f("fvecs");
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(-5.0f, 5.0f);
  const size_t dim = 7, count = 25;
  std::vector<float> data(dim * count);
  for (float& x : data) x = u(rng);

  write_fvecs(f.path(), data, dim);
  const FvecsData back = read_fvecs(f.path());
  CHECK_EQ(back.dim, dim);
  CHECK_EQ(back.count, count);
  CHECK(back.data == data);
}

static void test_ivecs_round_trip() {
  test::TempFile f("ivecs");
  std::vector<int32_t> data;
  for (int i = 0; i < 3 * 10; ++i) data.push_back(i * 1000003 - 7);
  write_ivecs(f.path(), data, 10);
  const IvecsData back = read_ivecs(f.path());
  CHECK_EQ(back.dim, size_t{10});
  CHECK_EQ(back.count, size_t{3});
  CHECK(back.data == data);
}

static void test_single_record_and_dim_one() {
  test::TempFile f("single");
  write_fvecs(f.path(), {1.5f}, 1);
  const FvecsData back = read_fvecs(f.path());
  CHECK_EQ(back.dim, size_t{1});
  CHECK_EQ(back.count, size_t{1});
  CHECK_EQ(back.data[0], 1.5f);
}

static void test_malformed_files() {
  test::TempFile f("bad");

  write_raw(f.path(), {});
  CHECK_THROWS(read_fvecs(f.path()), std::runtime_error, "empty");
  CHECK_THROWS(read_ivecs(f.path()), std::runtime_error, "empty");

  // 2 stray bytes: truncated dimension header.
  write_raw(f.path(), {}, 2);
  CHECK_THROWS(read_fvecs(f.path()), std::runtime_error, "truncated dimension header");

  // Header says 4 values but only 2 are present.
  write_raw(f.path(), {4, 0, 0});
  CHECK_THROWS(read_fvecs(f.path()), std::runtime_error, "truncated data in record 0");

  // Second record cut short.
  write_raw(f.path(), {2, 1, 2, 2, 3});
  CHECK_THROWS(read_ivecs(f.path()), std::runtime_error, "truncated data in record 1");

  // Complete record followed by 3 stray bytes.
  write_raw(f.path(), {2, 1, 2}, 3);
  CHECK_THROWS(read_ivecs(f.path()), std::runtime_error, "truncated dimension header in record 1");

  write_raw(f.path(), {0});
  CHECK_THROWS(read_fvecs(f.path()), std::runtime_error, "invalid dimension 0");

  write_raw(f.path(), {-3, 0, 0, 0});
  CHECK_THROWS(read_fvecs(f.path()), std::runtime_error, "invalid dimension -3");

  // Absurdly large dimension must fail cleanly, not allocate.
  write_raw(f.path(), {2000000000, 0});
  CHECK_THROWS(read_fvecs(f.path()), std::runtime_error, "truncated data in record 0");

  // Second record disagrees on dimension.
  write_raw(f.path(), {2, 1, 2, 3, 1, 2, 3});
  CHECK_THROWS(read_ivecs(f.path()), std::runtime_error, "record 1 has dimension 3, expected 2");

  CHECK_THROWS(read_fvecs(f.path() + ".missing"), std::runtime_error, "cannot open");
}

static void test_writer_validation() {
  test::TempFile f("writer");
  CHECK_THROWS(write_fvecs(f.path(), {1.0f, 2.0f}, 0), std::invalid_argument, "dim");
  CHECK_THROWS(write_fvecs(f.path(), {1.0f, 2.0f, 3.0f}, 2), std::invalid_argument, "multiple");
  CHECK_THROWS(write_fvecs(f.path(), {}, 2), std::invalid_argument, "multiple");
  CHECK_THROWS(write_ivecs(f.path(), {1, 2, 3}, 2), std::invalid_argument, "multiple");
  CHECK_THROWS(write_fvecs(f.path() + "/no/such/dir/x", {1.0f}, 1), std::runtime_error, "cannot open");
}

int main() {
  test_fvecs_round_trip();
  test_ivecs_round_trip();
  test_single_record_and_dim_one();
  test_malformed_files();
  test_writer_validation();
  return test_finish();
}
