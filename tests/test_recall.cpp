#include <cstdint>
#include <stdexcept>
#include <vector>

#include "hnsw/brute_force.h"
#include "test_helper.h"

using namespace hnsw;
using Ids = std::vector<uint32_t>;

static void test_basic() {
  CHECK_EQ(recall_at_k(Ids{1, 2, 3}, Ids{1, 2, 3}, 3), 1.0);
  CHECK_EQ(recall_at_k(Ids{3, 1, 2}, Ids{1, 2, 3}, 3), 1.0);  // order is irrelevant
  CHECK_EQ(recall_at_k(Ids{4, 5, 6}, Ids{1, 2, 3}, 3), 0.0);
  CHECK_EQ(recall_at_k(Ids{1, 9, 3, 8}, Ids{1, 2, 3, 4}, 4), 0.5);
  CHECK_NEAR(recall_at_k(Ids{1, 7, 8}, Ids{1, 2, 3}, 3), 1.0 / 3.0, 1e-12);
}

static void test_only_first_k_count() {
  // Entries beyond k in either list are ignored.
  CHECK_EQ(recall_at_k(Ids{1, 2, 9, 3}, Ids{1, 2, 7, 3}, 2), 1.0);
  CHECK_EQ(recall_at_k(Ids{9, 8, 1}, Ids{1, 2, 3}, 2), 0.0);
}

static void test_short_and_empty_lists() {
  CHECK_EQ(recall_at_k(Ids{1}, Ids{1, 2, 3, 4}, 4), 0.25);   // found shorter than k
  CHECK_EQ(recall_at_k(Ids{1, 2, 3, 4}, Ids{1, 2}, 4), 0.5); // truth shorter than k
  CHECK_EQ(recall_at_k(Ids{}, Ids{1, 2}, 2), 0.0);
  CHECK_EQ(recall_at_k(Ids{1, 2}, Ids{}, 2), 0.0);
  CHECK_EQ(recall_at_k(Ids{}, Ids{}, 5), 0.0);
}

static void test_duplicates_count_once() {
  CHECK_EQ(recall_at_k(Ids{1, 1, 1}, Ids{1, 2, 3}, 3), 1.0 / 3.0);
}

static void test_k_zero_throws() {
  CHECK_THROWS(recall_at_k(Ids{1}, Ids{1}, 0), std::invalid_argument, "k must be positive");
}

static void test_with_brute_force_truth() {
  const std::vector<float> base = {0, 1, 2, 3, 4, 5};  // dim 1
  const float q[] = {0.0f};
  const KnnResult truth = brute_force_knn(base.data(), 6, 1, q, 3);
  CHECK_EQ(recall_at_k(truth.ids, truth.ids, 3), 1.0);
}

int main() {
  test_basic();
  test_only_first_k_count();
  test_short_and_empty_lists();
  test_duplicates_count_once();
  test_k_zero_throws();
  test_with_brute_force_truth();
  return test_finish();
}
