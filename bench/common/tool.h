#pragma once

#include <chrono>
#include <cstdio>
#include <exception>

#include "env.h"

namespace bench {

using Clock = std::chrono::steady_clock;

inline double seconds_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// Runs a tool body: warns on stderr when the build is not a release build and
// turns exceptions into "NAME: error: ..." plus exit status 1.
template <typename F>
int run_tool(const char* name, F&& body) {
  if (!is_release_build()) {
    std::fprintf(stderr,
                 "%s: warning: not an optimized NDEBUG build without sanitizers; "
                 "timings are not representative\n",
                 name);
  }
  try {
    return body();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: error: %s\n", name, e.what());
    return 1;
  }
}

}  // namespace bench
