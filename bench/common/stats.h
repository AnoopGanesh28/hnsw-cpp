#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "json.h"

namespace bench {

struct Summary {
  double median = 0.0;
  double min = 0.0;
  double max = 0.0;
};

// Median, min and max of samples (in any order). For an even count the median
// is the mean of the two middle values. Throws std::invalid_argument if empty.
Summary summarize(const std::vector<double>& samples);

// {"raw": [samples in the order given], "median", "min", "max"}.
json::Value samples_json(const std::vector<double>& samples);

// 64-bit FNV-1a of `bytes` bytes at `data`, as 16 lowercase hex digits. Used
// to compare result arrays across processes and builds.
std::string fnv1a64_hex(const void* data, size_t bytes);

}  // namespace bench
