#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "hnsw/io.h"
#include "json.h"

namespace bench {

// Reads the first max_count records of an fvecs file with the same format
// checks as hnsw::read_fvecs, stopping there, so the rest of the file is never
// loaded into memory. kAllRecords reads the whole file. Throws
// std::runtime_error if the file is malformed or holds fewer than max_count
// records.
inline constexpr size_t kAllRecords = std::numeric_limits<size_t>::max();
hnsw::FvecsData read_fvecs_prefix(const std::string& path, size_t max_count);

// SHA-256 of a file, as 64 lowercase hex digits (computed with sha256sum).
std::string sha256_file(const std::string& path);

// One "file: NAME BYTES SHA256" line of a MANIFEST.
struct ManifestFile {
  uint64_t bytes = 0;
  std::string sha256;
};

// MANIFEST of a dataset directory, written by bench/get_sift.sh or
// bench_make_synthetic: "dataset: NAME", "file: ..." lines, and free-form
// "key: value" lines (kept in `fields`).
struct Manifest {
  std::string dir;  // absolute directory holding the MANIFEST
  std::string dataset;
  std::map<std::string, ManifestFile> files;
  std::map<std::string, std::string> fields;
};

// Reads DIR/MANIFEST, where DIR is the directory containing data_file.
// Throws std::runtime_error if it is missing or has no dataset line.
Manifest manifest_for(const std::string& data_file);

void write_manifest(const std::string& dir, const std::string& dataset,
                    const std::vector<std::pair<std::string, std::string>>& fields,
                    const std::vector<std::string>& file_names);

// {"path", "bytes", "sha256", "in_manifest"} for a file used by a benchmark.
// If the file is in manifest.dir and listed there, its size and checksum must match the
// listing, otherwise std::runtime_error is thrown. Files not listed (such as a
// ground truth recomputed for a subset) are recorded with in_manifest = false.
json::Value file_record(const std::string& path, const Manifest& manifest);

}  // namespace bench
