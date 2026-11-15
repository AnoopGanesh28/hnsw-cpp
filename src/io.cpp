#include "hnsw/io.h"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace hnsw {
namespace {

static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4,
              "fvecs/ivecs require 4-byte float and int32");

// The format is little-endian; values are copied byte-for-byte, so this
// requires a little-endian host (true for x86-64, the supported target).

template <typename T>
VecSet<T> read_vecs(const std::string& path, const char* kind) {
  const std::string who = std::string(kind) + " '" + path + "'";
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + who);
  }
  in.seekg(0, std::ios::end);
  const std::streamoff file_size = in.tellg();
  in.seekg(0, std::ios::beg);
  if (file_size <= 0) {
    throw std::runtime_error(who + ": file is empty");
  }

  VecSet<T> out;
  std::streamoff pos = 0;
  while (pos < file_size) {
    const size_t record = out.count;
    if (file_size - pos < 4) {
      throw std::runtime_error(who + ": truncated dimension header in record " +
                               std::to_string(record));
    }
    int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    pos += 4;
    if (d <= 0) {
      throw std::runtime_error(who + ": invalid dimension " + std::to_string(d) +
                               " in record " + std::to_string(record));
    }
    if (record == 0) {
      out.dim = static_cast<size_t>(d);
    } else if (static_cast<size_t>(d) != out.dim) {
      throw std::runtime_error(who + ": record " + std::to_string(record) +
                               " has dimension " + std::to_string(d) +
                               ", expected " + std::to_string(out.dim));
    }
    const std::streamoff payload =
        static_cast<std::streamoff>(out.dim) * static_cast<std::streamoff>(sizeof(T));
    if (file_size - pos < payload) {
      throw std::runtime_error(who + ": truncated data in record " +
                               std::to_string(record));
    }
    if (record == 0) {
      out.data.reserve(static_cast<size_t>((file_size - pos) / payload) * out.dim);
    }
    const size_t old = out.data.size();
    out.data.resize(old + out.dim);
    in.read(reinterpret_cast<char*>(out.data.data() + old),
            static_cast<std::streamsize>(payload));
    if (!in) {
      throw std::runtime_error(who + ": read error in record " +
                               std::to_string(record));
    }
    pos += payload;
    ++out.count;
  }
  return out;
}

template <typename T>
void write_vecs(const std::string& path, const std::vector<T>& data, size_t dim,
                const char* kind) {
  if (dim == 0 || dim > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument(std::string(kind) + ": dim must be in [1, INT32_MAX]");
  }
  if (data.empty() || data.size() % dim != 0) {
    throw std::invalid_argument(std::string(kind) +
                                ": data size must be a positive multiple of dim");
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot open for writing " + std::string(kind) +
                             " '" + path + "'");
  }
  const int32_t d = static_cast<int32_t>(dim);
  const size_t count = data.size() / dim;
  for (size_t i = 0; i < count; ++i) {
    out.write(reinterpret_cast<const char*>(&d), sizeof(d));
    out.write(reinterpret_cast<const char*>(data.data() + i * dim),
              static_cast<std::streamsize>(dim * sizeof(T)));
  }
  out.flush();
  if (!out) {
    throw std::runtime_error("write failed for " + std::string(kind) + " '" +
                             path + "'");
  }
}

}  // namespace

FvecsData read_fvecs(const std::string& path) {
  return read_vecs<float>(path, "fvecs");
}

IvecsData read_ivecs(const std::string& path) {
  return read_vecs<int32_t>(path, "ivecs");
}

void write_fvecs(const std::string& path, const std::vector<float>& data,
                 size_t dim) {
  write_vecs(path, data, dim, "fvecs");
}

void write_ivecs(const std::string& path, const std::vector<int32_t>& data,
                 size_t dim) {
  write_vecs(path, data, dim, "ivecs");
}

}  // namespace hnsw
