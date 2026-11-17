#pragma once

// AlignedBuffer: the growable 64-byte-aligned byte storage behind the layer-0 arena.

#include <cstddef>
#include <cstdint>

namespace hnsw {

// Growable byte buffer whose storage is 64-byte aligned. New bytes are zeroed.
// Growth is geometric, so repeated grow() calls are amortized O(1) per byte.
// Callers must hold offsets, not pointers, across calls that may grow it.
class AlignedBuffer {
 public:
  static constexpr size_t kAlign = 64;

  AlignedBuffer() = default;
  AlignedBuffer(const AlignedBuffer& other);
  AlignedBuffer(AlignedBuffer&& other) noexcept;
  AlignedBuffer& operator=(AlignedBuffer other) noexcept;
  ~AlignedBuffer();

  uint8_t* data() { return data_; }
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

  void reserve(size_t bytes);
  // Sets size to `bytes`; bytes added beyond the old size are zero.
  void resize(size_t bytes);

 private:
  void reallocate(size_t new_capacity);

  uint8_t* data_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
};

}  // namespace hnsw
