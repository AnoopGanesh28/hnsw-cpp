#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "hnsw/scratch.h"

namespace hnsw {

// Persistent pool of threads for data-parallel loops, reused across calls.
// Idle threads sleep on a condition variable (no busy waiting); the
// destructor wakes and joins them.
//
// The thread calling parallel_for takes part as thread index 0, so a pool of
// size() threads starts size() - 1 of its own. Each thread index owns one
// SearchScratch, reused across calls, for batch search.
class ThreadPool {
 public:
  // fn(thread_index, begin, end) processes items [begin, end).
  using Task = std::function<void(size_t thread_index, size_t begin, size_t end)>;

  // n_threads = 0 means std::thread::hardware_concurrency() (1 if unknown).
  explicit ThreadPool(size_t n_threads = 0);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  size_t size() const { return scratch_.size(); }

  // Calls fn over [0, n_items) in chunks of `chunk` items (the last may be
  // shorter). Threads claim chunks from a shared atomic counter, so faster
  // threads take more chunks. Returns when every claimed chunk has finished.
  // thread_index is in [0, size()) and is used by one thread at a time.
  //
  // If fn throws, no further chunks are started, and the first exception is
  // rethrown here once running chunks finish. Calls from several threads are
  // serialized. fn must not call parallel_for on the same pool (deadlock).
  // Throws std::invalid_argument if chunk == 0.
  void parallel_for(size_t n_items, size_t chunk, const Task& fn);

  // Scratch owned by thread_index. Inside parallel_for only that thread index
  // may use it; outside, do not use it while a parallel_for is running.
  SearchScratch& scratch(size_t thread_index) { return scratch_[thread_index]; }

 private:
  void worker(size_t thread_index);
  void run_chunks(size_t thread_index);
  void stop_workers() noexcept;

  std::vector<SearchScratch> scratch_;
  std::vector<std::thread> workers_;

  std::mutex call_mutex_;  // serializes parallel_for callers

  std::mutex mutex_;  // guards everything below except next_
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  uint64_t generation_ = 0;  // incremented once per parallel_for
  bool stop_ = false;
  size_t pending_ = 0;  // workers still running the current generation
  const Task* task_ = nullptr;
  size_t n_items_ = 0;
  size_t chunk_ = 0;
  std::exception_ptr error_;

  std::atomic<size_t> next_{0};  // first unclaimed item
};

}  // namespace hnsw
