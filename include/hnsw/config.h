#pragma once

// Compile-time optimization switches, set by CMake (HNSW_OPT_* options) as
// public definitions of the hnsw target so the library and its users agree.
#if !defined(HNSW_OPT_EPOCH_VISITED) || !defined(HNSW_OPT_ALLOC_FREE_HEAPS)
#error "HNSW_OPT_* macros are defined by the hnsw CMake target; link against it"
#endif

namespace hnsw {

inline constexpr bool kEpochVisited = HNSW_OPT_EPOCH_VISITED != 0;
inline constexpr bool kAllocFreeHeaps = HNSW_OPT_ALLOC_FREE_HEAPS != 0;

}  // namespace hnsw
