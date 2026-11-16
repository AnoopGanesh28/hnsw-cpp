// ThreadPool: coverage of every item exactly once, dynamic chunking, real
// concurrency, exception propagation, reuse, and clean shutdown. Also run
// under ThreadSanitizer (HNSW_SANITIZE=thread).

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hnsw/thread_pool.h"
#include "test_helper.h"

using hnsw::ThreadPool;

// Every item is visited exactly once, in chunks of the requested size, by
// valid thread indices.
static void test_covers_each_item_once() {
  for (size_t threads : {1u, 2u, 3u, 8u}) {
    ThreadPool pool(threads);
    CHECK_EQ(pool.size(), threads);
    for (size_t n : {1u, 2u, 7u, 100u, 1000u}) {
      for (size_t chunk : {1u, 3u, 64u, 5000u}) {
        std::vector<std::atomic<int>> hits(n);
        std::atomic<bool> bad_chunk{false}, bad_index{false};
        pool.parallel_for(n, chunk, [&](size_t t, size_t b, size_t e) {
          if (t >= threads) bad_index = true;
          if (b >= e || e > n || (e - b != chunk && e != n) || b % chunk != 0) bad_chunk = true;
          for (size_t i = b; i < e; ++i) hits[i].fetch_add(1);
        });
        bool once = true;
        for (const auto& h : hits) once = once && h.load() == 1;
        CHECK(once);
        CHECK(!bad_chunk.load());
        CHECK(!bad_index.load());
      }
    }
  }
}

static void test_empty_and_invalid() {
  ThreadPool pool(4);
  bool called = false;
  pool.parallel_for(0, 1, [&](size_t, size_t, size_t) { called = true; });
  CHECK(!called);
  CHECK_THROWS(pool.parallel_for(10, 0, [](size_t, size_t, size_t) {}), std::invalid_argument,
               "chunk");
}

static void test_default_size() {
  const ThreadPool pool(0);
  const size_t hc = std::thread::hardware_concurrency();
  CHECK_EQ(pool.size(), hc == 0 ? size_t{1} : hc);
}

// The thread that takes item 0 blocks until another thread has processed
// item 1, which only terminates if two threads really run at once.
static void test_runs_concurrently() {
  ThreadPool pool(2);
  std::mutex m;
  std::condition_variable cv;
  bool item1_done = false;
  std::atomic<int> threads_seen_mask{0};
  pool.parallel_for(2, 1, [&](size_t t, size_t b, size_t) {
    threads_seen_mask.fetch_or(1 << t);
    std::unique_lock<std::mutex> lock(m);
    if (b == 1) {
      item1_done = true;
      cv.notify_all();
    } else {
      cv.wait(lock, [&] { return item1_done; });
    }
  });
  CHECK_EQ(threads_seen_mask.load(), 3);
}

static void test_exception_propagates_and_pool_survives() {
  for (size_t threads : {1u, 4u}) {
    ThreadPool pool(threads);
    std::atomic<size_t> processed{0};
    CHECK_THROWS(pool.parallel_for(1000, 1,
                                   [&](size_t, size_t b, size_t) {
                                     if (b == 37) throw std::runtime_error("item 37 failed");
                                     processed.fetch_add(1);
                                   }),
                 std::runtime_error, "item 37 failed");
    // Single-threaded, chunks run in order and stop at the failure.
    if (threads == 1) CHECK_EQ(processed.load(), size_t{37});
    // Every chunk throws: exactly one exception reaches the caller.
    CHECK_THROWS(pool.parallel_for(100, 1,
                                   [](size_t, size_t, size_t) { throw std::logic_error("all"); }),
                 std::logic_error, "all");
    std::atomic<size_t> sum{0};
    pool.parallel_for(1000, 7, [&](size_t, size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) sum.fetch_add(i);
    });
    CHECK_EQ(sum.load(), size_t{999 * 1000 / 2});
  }
}

static void test_reuse_across_many_batches() {
  ThreadPool pool(4);
  std::vector<size_t> out(64);
  bool ok = true;
  for (size_t round = 0; round < 3000; ++round) {
    const size_t n = 1 + round % 64;
    pool.parallel_for(n, 1 + round % 5, [&](size_t, size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) out[i] = round * 1000 + i;
    });
    for (size_t i = 0; i < n; ++i) ok = ok && out[i] == round * 1000 + i;
  }
  CHECK(ok);
}

// Several user threads share one pool; their calls are serialized and each
// gets its own complete result.
static void test_concurrent_callers() {
  ThreadPool pool(3);
  std::vector<std::thread> users;
  std::atomic<int> wrong{0};
  for (int u = 0; u < 4; ++u) {
    users.emplace_back([&, u] {
      for (int round = 0; round < 50; ++round) {
        std::vector<int> v(200, 0);
        pool.parallel_for(v.size(), 4, [&](size_t, size_t b, size_t e) {
          for (size_t i = b; i < e; ++i) v[i] = u + round;
        });
        for (int x : v) wrong += x != u + round;
      }
    });
  }
  for (std::thread& t : users) t.join();
  CHECK_EQ(wrong.load(), 0);
}

static void test_clean_destruction() {
  for (int i = 0; i < 50; ++i) {
    ThreadPool idle(1 + i % 9);  // destroyed without ever running work
  }
  for (int i = 0; i < 20; ++i) {
    ThreadPool used(6);
    std::atomic<int> c{0};
    used.parallel_for(10, 1, [&](size_t, size_t, size_t) { c.fetch_add(1); });
    CHECK_EQ(c.load(), 10);
  }  // destroyed with idle workers after a batch
}

int main() {
  test_covers_each_item_once();
  test_empty_and_invalid();
  test_default_size();
  test_runs_concurrently();
  test_exception_propagates_and_pool_survives();
  test_reuse_across_many_batches();
  test_concurrent_callers();
  test_clean_destruction();
  return test_finish();
}
