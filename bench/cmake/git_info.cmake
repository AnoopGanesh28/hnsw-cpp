# Writes OUT (a header) with the git state of SRC and HNSWLIB. Run at every
# build by the bench_git_info target; OUT is only rewritten when it changes,
# so unchanged state recompiles nothing.
# Inputs: -DGIT=<git executable> -DSRC=<repo> -DHNSWLIB=<hnswlib checkout> -DOUT=<header>
function(git_query out_var dir)
  execute_process(COMMAND "${GIT}" -C "${dir}" ${ARGN}
                  OUTPUT_VARIABLE value RESULT_VARIABLE rc
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(NOT rc EQUAL 0)
    set(value "unknown")
  endif()
  set(${out_var} "${value}" PARENT_SCOPE)
endfunction()

git_query(commit "${SRC}" rev-parse HEAD)
git_query(describe "${SRC}" describe --tags --always --dirty)
git_query(status "${SRC}" status --porcelain --untracked-files=no)
if(status STREQUAL "")
  set(dirty "false")
elseif(status STREQUAL "unknown")
  set(dirty "unknown")
else()
  set(dirty "true")
endif()
git_query(hnswlib_commit "${HNSWLIB}" rev-parse HEAD)

set(content "#pragma once
// Generated at build time by bench/cmake/git_info.cmake. Do not edit.
#define HNSW_BENCH_GIT_COMMIT \"${commit}\"
#define HNSW_BENCH_GIT_DESCRIBE \"${describe}\"
#define HNSW_BENCH_GIT_DIRTY \"${dirty}\"
#define HNSW_BENCH_HNSWLIB_COMMIT \"${hnswlib_commit}\"
")
file(WRITE "${OUT}.tmp" "${content}")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${OUT}.tmp" "${OUT}")
file(REMOVE "${OUT}.tmp")
