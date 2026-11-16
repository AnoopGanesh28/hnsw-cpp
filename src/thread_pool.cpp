#include "hnsw/thread_pool.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace hnsw {

namespace {
size_t resolve_threads(size_t n) {
  if (n != 0) return n;
  return std::max<size_t>(1, std::thread::hardware_concurrency());
}
}  // namespace

ThreadPool::ThreadPool(size_t n_threads) : scratch_(resolve_threads(n_threads)) {
  workers_.reserve(scratch_.size() - 1);
  try {
    for (size_t i = 1; i < scratch_.size(); ++i) workers_.emplace_back([this, i] { worker(i); });
  } catch (...) {
    stop_workers();
    throw;
  }
}

ThreadPool::~ThreadPool() { stop_workers(); }

void ThreadPool::stop_workers() noexcept {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  work_cv_.notify_all();
  for (std::thread& t : workers_) t.join();
  workers_.clear();
}

void ThreadPool::worker(size_t thread_index) {
  uint64_t seen = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      work_cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
    }
    run_chunks(thread_index);
    std::lock_guard<std::mutex> lock(mutex_);
    if (--pending_ == 0) done_cv_.notify_one();
  }
}

// Claims chunks until none are left. chunk_ <= n_items_, so the counter
// stays below n_items_ * (size() + 1) and cannot wrap for any real n_items.
void ThreadPool::run_chunks(size_t thread_index) {
  for (;;) {
    const size_t begin = next_.fetch_add(chunk_, std::memory_order_relaxed);
    if (begin >= n_items_) return;
    const size_t end = begin + std::min(chunk_, n_items_ - begin);
    try {
      (*task_)(thread_index, begin, end);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!error_) error_ = std::current_exception();
      next_.store(n_items_, std::memory_order_relaxed);
    }
  }
}

void ThreadPool::parallel_for(size_t n_items, size_t chunk, const Task& fn) {
  if (chunk == 0) throw std::invalid_argument("ThreadPool::parallel_for: chunk must be >= 1");
  if (n_items == 0) return;
  std::lock_guard<std::mutex> call(call_mutex_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    task_ = &fn;
    n_items_ = n_items;
    chunk_ = std::min(chunk, n_items);
    next_.store(0, std::memory_order_relaxed);
    error_ = nullptr;
    pending_ = workers_.size();
    ++generation_;
  }
  work_cv_.notify_all();
  run_chunks(0);
  std::exception_ptr error;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [&] { return pending_ == 0; });
    error = std::exchange(error_, nullptr);
    task_ = nullptr;
  }
  if (error) std::rethrow_exception(error);
}

}  // namespace hnsw
