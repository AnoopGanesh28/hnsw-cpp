#include "args.h"

#include <cerrno>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace bench {

namespace {

size_t parse_size(const std::string& name, const std::string& text) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("--" + name + ": expected a non-negative integer, got '" +
                                text + "'");
  }
  errno = 0;
  const unsigned long long v = std::strtoull(text.c_str(), nullptr, 10);
  if (errno != 0) throw std::invalid_argument("--" + name + ": value out of range");
  return static_cast<size_t>(v);
}

std::vector<std::string> split(const std::string& text) {
  std::vector<std::string> out;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(item);
  return out;
}

}  // namespace

Args::Args(int argc, char** argv, int first) {
  for (int i = first; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.size() < 3 || a.compare(0, 2, "--") != 0) {
      throw std::invalid_argument("unexpected argument '" + a + "' (flags are --name value)");
    }
    if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
    const std::string name = a.substr(2);
    if (!values_.emplace(name, argv[++i]).second) {
      throw std::invalid_argument(a + " given more than once");
    }
  }
}

bool Args::has(const std::string& name) {
  known_.insert(name);
  return values_.count(name) != 0;
}

std::string Args::str(const std::string& name) {
  if (!has(name)) throw std::invalid_argument("missing required flag --" + name);
  return values_.at(name);
}

std::string Args::str(const std::string& name, const std::string& fallback) {
  return has(name) ? values_.at(name) : fallback;
}

size_t Args::size(const std::string& name) { return parse_size(name, str(name)); }

size_t Args::size(const std::string& name, size_t fallback) {
  return has(name) ? parse_size(name, values_.at(name)) : fallback;
}

double Args::real(const std::string& name, double fallback) {
  if (!has(name)) return fallback;
  const std::string& t = values_.at(name);
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(t.c_str(), &end);
  if (t.empty() || end != t.c_str() + t.size() || errno != 0) {
    throw std::invalid_argument("--" + name + ": expected a number, got '" + t + "'");
  }
  return v;
}

std::vector<size_t> Args::sizes(const std::string& name, const std::vector<size_t>& fallback) {
  if (!has(name)) return fallback;
  std::vector<size_t> out;
  for (const std::string& item : split(values_.at(name))) out.push_back(parse_size(name, item));
  if (out.empty()) throw std::invalid_argument("--" + name + ": empty list");
  return out;
}

std::vector<std::string> Args::words(const std::string& name,
                                     const std::vector<std::string>& fallback) {
  if (!has(name)) return fallback;
  std::vector<std::string> out = split(values_.at(name));
  if (out.empty()) throw std::invalid_argument("--" + name + ": empty list");
  for (const std::string& w : out) {
    if (w.empty()) throw std::invalid_argument("--" + name + ": empty list item");
  }
  return out;
}

std::string Args::choice(const std::string& name, const std::vector<std::string>& choices,
                         const std::string& fallback) {
  const std::string v = fallback.empty() ? str(name) : str(name, fallback);
  for (const std::string& c : choices) {
    if (c == v) return v;
  }
  std::string all;
  for (const std::string& c : choices) all += (all.empty() ? "" : ", ") + c;
  throw std::invalid_argument("--" + name + " must be one of: " + all + " (got '" + v + "')");
}

void Args::finish() const {
  for (const auto& kv : values_) {
    if (!known_.count(kv.first)) throw std::invalid_argument("unknown flag --" + kv.first);
  }
}

}  // namespace bench
