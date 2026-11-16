# Benchmark tools (HNSW_BUILD_BENCH=ON). They compile with exactly the flags
# of the library (inherited directory options, CMAKE_CXX_FLAGS_<CONFIG>), and
# hnswlib's headers are compiled with those same flags.

set(HNSWLIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib)

# The pin lives in get_hnswlib.sh only.
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/get_hnswlib.sh _pin REGEX "^HNSWLIB_(TAG|COMMIT)=")
foreach(_line IN LISTS _pin)
  if(_line MATCHES "^(HNSWLIB_TAG|HNSWLIB_COMMIT)=(.+)$")
    set(${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
  endif()
endforeach()
if(NOT HNSWLIB_TAG OR NOT HNSWLIB_COMMIT)
  message(FATAL_ERROR "bench/get_hnswlib.sh must define HNSWLIB_TAG and HNSWLIB_COMMIT")
endif()

if(NOT EXISTS ${HNSWLIB_DIR}/hnswlib/hnswlib.h)
  message(FATAL_ERROR
    "HNSW_BUILD_BENCH=ON needs the hnswlib baseline, which is not in ${HNSWLIB_DIR}.\n"
    "Run bench/get_hnswlib.sh (clones hnswlib ${HNSWLIB_TAG}), then configure again, "
    "or configure with -DHNSW_BUILD_BENCH=OFF.")
endif()

find_package(Git REQUIRED)
execute_process(COMMAND ${GIT_EXECUTABLE} -C ${HNSWLIB_DIR} rev-parse HEAD
                OUTPUT_VARIABLE _hnswlib_head OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _rc ERROR_QUIET)
if(NOT _rc EQUAL 0 OR NOT _hnswlib_head STREQUAL HNSWLIB_COMMIT)
  message(FATAL_ERROR
    "${HNSWLIB_DIR} is at '${_hnswlib_head}', but the pinned baseline is "
    "${HNSWLIB_TAG} (${HNSWLIB_COMMIT}). Remove it and run bench/get_hnswlib.sh.")
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT HNSW_SANITIZE STREQUAL "off")
  message(WARNING "Benchmark tools in a ${CMAKE_BUILD_TYPE} build with HNSW_SANITIZE="
                  "${HNSW_SANITIZE}: fine for tests, not for measurements.")
endif()

# Configure-time record of the toolchain and the effective common flags.
function(first_line_of_command out_var)
  execute_process(COMMAND ${ARGN} OUTPUT_VARIABLE text RESULT_VARIABLE rc ERROR_QUIET)
  if(rc EQUAL 0)
    string(REGEX MATCH "^[^\n]*" text "${text}")
  else()
    set(text "not found")
  endif()
  set(${out_var} "${text}" PARENT_SCOPE)
endfunction()
first_line_of_command(HNSW_BENCH_GXX_VERSION g++ --version)
first_line_of_command(HNSW_BENCH_CLANGXX_VERSION clang++ --version)

string(TOUPPER "${CMAKE_BUILD_TYPE}" _config)
get_directory_property(_dir_options COMPILE_OPTIONS)
list(JOIN _dir_options " " _dir_options)
set(HNSW_BENCH_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_config}} ${CMAKE_CXX17_STANDARD_COMPILE_OPTION} ${_dir_options}")
string(STRIP "${HNSW_BENCH_CXX_FLAGS}" HNSW_BENCH_CXX_FLAGS)
string(REGEX REPLACE " +" " " HNSW_BENCH_CXX_FLAGS "${HNSW_BENCH_CXX_FLAGS}")
list(JOIN HNSW_AVX2_FILE_OPTIONS " " HNSW_BENCH_AVX2_FILE_OPTIONS)
list(JOIN HNSW_STRICT_FP " " HNSW_BENCH_SCALAR_FILE_OPTIONS)

set(_gen ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(build_info.h.in ${_gen}/build_info.h @ONLY)

# Build-time record of the git state, refreshed on every build.
add_custom_target(bench_git_info
  COMMAND ${CMAKE_COMMAND} -DGIT=${GIT_EXECUTABLE} -DSRC=${PROJECT_SOURCE_DIR}
          -DHNSWLIB=${HNSWLIB_DIR} -DOUT=${_gen}/git_info.h
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_info.cmake
  BYPRODUCTS ${_gen}/git_info.h
  COMMENT "Recording git state for benchmark results")

add_library(hnsw_bench STATIC
  common/args.cpp
  common/dataset.cpp
  common/env.cpp
  common/exact.cpp
  common/hnswlib_index.cpp
  common/json.cpp
  common/proc.cpp
  common/stats.cpp)
add_dependencies(hnsw_bench bench_git_info)
# hnswlib v0.9.0 over-reads one id past the end of a neighbor list to compute
# a prefetch address (hnswalg.h, searchBaseLayer: *(datal + j + 1)); ASan
# rightly aborts on it. The baseline is not ours to patch, so in ASan builds
# its one translation unit is compiled without ASan (UBSan stays on).
if(HNSW_SANITIZE MATCHES "address")
  set_source_files_properties(common/hnswlib_index.cpp PROPERTIES
                              COMPILE_OPTIONS "-fno-sanitize=address")
endif()
target_include_directories(hnsw_bench PUBLIC common PRIVATE ${_gen})
# SYSTEM: hnswlib's own warnings are not ours to fix.
target_include_directories(hnsw_bench SYSTEM PRIVATE ${HNSWLIB_DIR})
target_link_libraries(hnsw_bench PUBLIC hnsw)

add_executable(bench_build bench_build.cpp)
add_executable(bench_search bench_search.cpp)
add_executable(bench_loadtime bench_loadtime.cpp)
add_executable(bench_gt bench_gt.cpp)
add_executable(bench_make_synthetic make_synthetic.cpp)
foreach(_tool bench_build bench_search bench_loadtime bench_gt bench_make_synthetic)
  target_link_libraries(${_tool} PRIVATE hnsw_bench)
endforeach()

# Tests (sources in tests/, registered only in bench builds).
add_executable(test_bench_util ${PROJECT_SOURCE_DIR}/tests/test_bench_util.cpp)
target_link_libraries(test_bench_util PRIVATE hnsw_bench)
target_include_directories(test_bench_util PRIVATE ${PROJECT_SOURCE_DIR}/tests)
add_test(NAME test_bench_util COMMAND test_bench_util)

add_executable(test_bench_smoke ${PROJECT_SOURCE_DIR}/tests/test_bench_smoke.cpp)
target_link_libraries(test_bench_smoke PRIVATE hnsw_bench)
target_include_directories(test_bench_smoke PRIVATE ${PROJECT_SOURCE_DIR}/tests)
add_dependencies(test_bench_smoke
                 bench_build bench_search bench_loadtime bench_gt bench_make_synthetic)
add_test(NAME test_bench_smoke COMMAND test_bench_smoke $<TARGET_FILE_DIR:bench_build>)
