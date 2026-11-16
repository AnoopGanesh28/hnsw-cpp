#pragma once

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

}  // namespace bench
