#pragma once

// Replaces every global operator new / delete form with a version that
// counts allocations and records the largest single request. Include from
// exactly one translation unit of a test executable.

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace test {

inline std::atomic<long> g_allocations{0};
inline std::atomic<size_t> g_max_allocation{0};

inline void* counted_alloc(std::size_t size, std::size_t align) {
  ++g_allocations;
  size_t prev = g_max_allocation.load();
  while (size > prev && !g_max_allocation.compare_exchange_weak(prev, size)) {
  }
  if (size == 0) size = 1;
  void* p = align > alignof(std::max_align_t)
                ? std::aligned_alloc(align, (size + align - 1) / align * align)
                : std::malloc(size);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}

}  // namespace test

void* operator new(std::size_t n) { return test::counted_alloc(n, 0); }
void* operator new[](std::size_t n) { return test::counted_alloc(n, 0); }
void* operator new(std::size_t n, std::align_val_t a) {
  return test::counted_alloc(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) {
  return test::counted_alloc(n, static_cast<std::size_t>(a));
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  try { return test::counted_alloc(n, 0); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  try { return test::counted_alloc(n, 0); } catch (...) { return nullptr; }
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  try { return test::counted_alloc(n, static_cast<std::size_t>(a)); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  try { return test::counted_alloc(n, static_cast<std::size_t>(a)); } catch (...) { return nullptr; }
}
// Every form frees with std::free, matching counted_alloc. GCC cannot see
// that operator new above uses malloc and, after inlining, may warn that free
// receives memory from operator new.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
