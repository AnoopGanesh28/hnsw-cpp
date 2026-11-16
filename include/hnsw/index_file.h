#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "hnsw/brute_force.h"
#include "hnsw/distance.h"
#include "hnsw/graph.h"
#include "hnsw/layout.h"
#include "hnsw/scratch.h"

namespace hnsw {

// On-disk index format, version 1. The byte-level specification is
// docs/FORMAT.md. Files are written by HnswIndex::save and read by
// LoadedIndex::open; the layer-0 section holds the in-memory records of
// layout.h verbatim, so a mapped file is searched in place.
namespace format {
inline constexpr uint32_t kVersion = 1;
inline constexpr size_t kHeaderSize = 192;
inline constexpr size_t kSectionAlign = 64;
// Limits enforced by both save and load.
inline constexpr size_t kMaxDim = size_t{1} << 16;
inline constexpr size_t kMaxM = 1024;
}  // namespace format

// Thrown for every failure to write or read an index file. what() names the
// file and what was wrong.
class IndexFileError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class LoadMode {
  Mmap,  // map the file read-only (PROT_READ, MAP_PRIVATE); search runs on the mapping
  Read,  // read the whole file into a 64-byte-aligned heap buffer
};

struct LoadOptions {
  LoadMode mode = LoadMode::Mmap;
  // Validates the whole graph and every vector on open (O(file size)).
  //
  // With validate_deep = true, searching any file that open() accepts is
  // memory-safe, whatever its bytes. With validate_deep = false only the
  // header and section bounds are checked (O(1)); link counts, link ids and
  // upper-layer offsets are trusted, so a corrupt or malicious file can make
  // search read out of bounds. Turn it off only for files you trust.
  bool validate_deep = true;
  // Distance kernel used for search. Must compute the file's metric
  // (squared L2, the only metric of format version 1).
  DistFn dist = best_l2_squared();
};

// Read-only index loaded from a file written by HnswIndex::save. Searches use
// the same GraphView code path as HnswIndex and, for the same kernel, return
// bit-identical results. Move-only; the mapping or buffer is released in the
// destructor (the file descriptor is closed before open() returns).
//
// In Mmap mode the file must not be truncated or modified while loaded
// (validation happens once, in open()). Replacing it with HnswIndex::save is
// safe: save renames a new file into place and the old mapping keeps the old
// contents.
//
// Searching is const and the graph is never written, so any number of threads
// may search one LoadedIndex concurrently, each with its own SearchScratch.
class LoadedIndex {
 public:
  // Throws IndexFileError if the file cannot be read or fails validation,
  // std::invalid_argument if options.dist is null.
  static LoadedIndex open(const std::string& path, const LoadOptions& options = {});

  LoadedIndex(LoadedIndex&& other) noexcept;
  LoadedIndex& operator=(LoadedIndex&& other) noexcept;
  LoadedIndex(const LoadedIndex&) = delete;
  LoadedIndex& operator=(const LoadedIndex&) = delete;
  ~LoadedIndex();

  const GraphView& view() const { return view_; }
  LoadMode mode() const { return mode_; }
  size_t size() const { return view_.count; }
  size_t dim() const { return view_.layout.dim; }
  size_t M() const { return view_.layout.M; }
  size_t ef_construction() const { return ef_construction_; }
  const Layout& layout() const { return view_.layout; }
  // Total file size in bytes.
  size_t file_size() const { return bytes_; }

  // Same contracts as the HnswIndex accessors of the same names.
  int max_level() const { return view_.max_level; }
  uint32_t entry_point() const;
  int node_level(uint32_t id) const;
  NeighborView neighbors(uint32_t id, int level) const;
  const float* vector(uint32_t id) const;

  // Same contracts as HnswIndex::search / search_into.
  KnnResult search(const float* query, size_t k, size_t ef) const;
  KnnResult search(const float* query, size_t k, size_t ef, SearchScratch& scratch) const;
  size_t search_into(const float* query, size_t k, size_t ef, SearchScratch& scratch,
                     uint32_t* out_ids, float* out_dists) const;

 private:
  struct AlignedFree {
    void operator()(uint8_t* p) const;
  };

  LoadedIndex() = default;
  void release() noexcept;

  GraphView view_;
  DistFn dist_ = nullptr;
  size_t ef_construction_ = 0;
  LoadMode mode_ = LoadMode::Mmap;
  void* map_ = nullptr;  // Mmap mode: the mapping, bytes_ long
  std::unique_ptr<uint8_t, AlignedFree> buffer_;  // Read mode
  size_t bytes_ = 0;
};

}  // namespace hnsw
