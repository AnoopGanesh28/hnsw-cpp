#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace bench {

// Command-line flags of the form "--name value". Every flag takes a value and
// may appear once. Getters mark a flag as known; finish() then rejects any
// flag that no getter asked for, so typos fail instead of being ignored.
// All errors throw std::invalid_argument with a message naming the flag.
class Args {
 public:
  // Parses argv[first..argc).
  Args(int argc, char** argv, int first = 1);

  bool has(const std::string& name);
  std::string str(const std::string& name);  // required
  std::string str(const std::string& name, const std::string& fallback);
  size_t size(const std::string& name);  // required, non-negative integer
  size_t size(const std::string& name, size_t fallback);
  double real(const std::string& name, double fallback);
  // Comma-separated non-negative integers, e.g. "10,20,40".
  std::vector<size_t> sizes(const std::string& name, const std::vector<size_t>& fallback);
  // Comma-separated words.
  std::vector<std::string> words(const std::string& name,
                                 const std::vector<std::string>& fallback);
  // Value must be one of `choices`; an empty fallback makes the flag required.
  std::string choice(const std::string& name, const std::vector<std::string>& choices,
                     const std::string& fallback);

  void finish() const;

 private:
  std::map<std::string, std::string> values_;
  std::set<std::string> known_;
};

}  // namespace bench
