#include "hnsw/aligned_buffer.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

namespace hnsw {

AlignedBuffer::AlignedBuffer(const AlignedBuffer& other) {
  reallocate(other.size_);
  if (other.size_ > 0) std::memcpy(data_, other.data_, other.size_);
  size_ = other.size_;
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      capacity_(std::exchange(other.capacity_, 0)) {}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer other) noexcept {
  std::swap(data_, other.data_);
  std::swap(size_, other.size_);
  std::swap(capacity_, other.capacity_);
  return *this;
}

AlignedBuffer::~AlignedBuffer() { std::free(data_); }

void AlignedBuffer::reallocate(size_t new_capacity) {
  if (new_capacity == 0) return;
  const size_t bytes = (new_capacity + kAlign - 1) / kAlign * kAlign;
  auto* fresh = static_cast<uint8_t*>(std::aligned_alloc(kAlign, bytes));
  if (fresh == nullptr) throw std::bad_alloc();
  if (size_ > 0) std::memcpy(fresh, data_, size_);
  std::free(data_);
  data_ = fresh;
  capacity_ = bytes;
}

void AlignedBuffer::reserve(size_t bytes) {
  if (bytes > capacity_) reallocate(bytes);
}

void AlignedBuffer::resize(size_t bytes) {
  if (bytes > capacity_) reallocate(std::max(bytes, 2 * capacity_));
  if (bytes > size_) std::memset(data_ + size_, 0, bytes - size_);
  size_ = bytes;
}

}  // namespace hnsw
