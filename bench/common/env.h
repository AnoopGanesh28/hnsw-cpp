#pragma once

#include <cstddef>

#include "json.h"

namespace bench {

// The environment block written into every result file: date, CPU, memory,
// kernel, WSL detection, the head of lscpu, compilers and the exact flags of
// this build, the git state of this repository and of hnswlib at build time,
// which distance kernels both systems use for vectors of dimension `dim`, and
// whether this is an optimized, assertion-free, unsanitized build.
json::Value environment_record(size_t dim);

// False unless this binary was built optimized, with NDEBUG and without
// sanitizers; tools print a warning when it is false.
bool is_release_build();

}  // namespace bench
