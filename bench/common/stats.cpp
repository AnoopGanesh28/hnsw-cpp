#include "stats.h"

#include <algorithm>
#include <stdexcept>

namespace bench {

Summary summarize(const std::vector<double>& samples) {
  if (samples.empty()) throw std::invalid_argument("summarize: no samples");
  std::vector<double> s = samples;
  std::sort(s.begin(), s.end());
  const size_t n = s.size();
  Summary out;
  out.median = n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2.0;
  out.min = s.front();
  out.max = s.back();
  return out;
}

json::Value samples_json(const std::vector<double>& samples) {
  const Summary s = summarize(samples);
  json::Value out = json::Value::object();
  out.set("raw", json::Value::array_of(samples));
  out.set("median", s.median);
  out.set("min", s.min);
  out.set("max", s.max);
  return out;
}

}  // namespace bench
