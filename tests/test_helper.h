#pragma once

// Minimal dependency-free test helper. Each test file defines main(), calls
// its test functions, and returns test_finish(). Failed checks print
// file:line and are counted; the process exits nonzero if any failed.

#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace test {

inline int& checks() { static int n = 0; return n; }
inline int& failures() { static int n = 0; return n; }

inline void fail(const char* file, int line, const std::string& msg) {
  ++failures();
  std::fprintf(stderr, "%s:%d: FAILED: %s\n", file, line, msg.c_str());
}

// Temp file path removed on destruction.
class TempFile {
 public:
  explicit TempFile(const std::string& tag) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("hnsw_test_" + tag + "_" + std::to_string(::getpid()) + "_" +
              std::to_string(counter++)))
                .string();
  }
  ~TempFile() { std::error_code ec; std::filesystem::remove(path_, ec); }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

}  // namespace test

#define CHECK(cond)                                                   \
  do {                                                                \
    ++test::checks();                                                 \
    if (!(cond)) test::fail(__FILE__, __LINE__, "CHECK(" #cond ")");  \
  } while (0)

#define CHECK_EQ(a, b)                                                       \
  do {                                                                       \
    ++test::checks();                                                        \
    if (!((a) == (b)))                                                       \
      test::fail(__FILE__, __LINE__, "CHECK_EQ(" #a ", " #b ")");            \
  } while (0)

#define CHECK_NEAR(a, b, eps)                                                \
  do {                                                                       \
    ++test::checks();                                                        \
    const double a_ = static_cast<double>(a), b_ = static_cast<double>(b);   \
    if (!(std::fabs(a_ - b_) <= static_cast<double>(eps)))                   \
      test::fail(__FILE__, __LINE__,                                         \
                 "CHECK_NEAR(" #a ", " #b ", " #eps ") got " +               \
                     std::to_string(a_) + " vs " + std::to_string(b_));      \
  } while (0)

// Passes if expr throws an exception of type ExType whose what() contains substr.
#define CHECK_THROWS(expr, ExType, substr)                                   \
  do {                                                                       \
    ++test::checks();                                                        \
    try {                                                                    \
      (void)(expr);                                                          \
      test::fail(__FILE__, __LINE__, "no exception from " #expr);            \
    } catch (const ExType& e_) {                                             \
      if (std::string(e_.what()).find(substr) == std::string::npos)          \
        test::fail(__FILE__, __LINE__,                                       \
                   std::string("wrong message from " #expr ": ") + e_.what()); \
    } catch (...) {                                                          \
      test::fail(__FILE__, __LINE__, "wrong exception type from " #expr);    \
    }                                                                        \
  } while (0)

inline int test_finish() {
  std::fprintf(stderr, "%d checks, %d failed\n", test::checks(), test::failures());
  return test::failures() == 0 ? 0 : 1;
}
