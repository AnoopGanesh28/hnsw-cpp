cmake_minimum_required(VERSION 3.16)
project(hnsw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(HNSW_NATIVE "Compile with -march=native" ON)
set(HNSW_SANITIZE "off" CACHE STRING
    "Sanitizers: off, address, undefined, or address,undefined")
set_property(CACHE HNSW_SANITIZE PROPERTY STRINGS
             off address undefined "address,undefined")

option(HNSW_OPT_EPOCH_VISITED "Epoch-tagged visited list in SearchScratch" ON)
option(HNSW_OPT_ALLOC_FREE_HEAPS "Fixed-capacity search heaps in SearchScratch" ON)
option(HNSW_OPT_PREFETCH "Software prefetch in the neighbor-expansion loop" ON)

add_compile_options(-Wall -Wextra)
# Strict IEEE evaluation of scalar code: forbids the compiler from fusing a*b+c
# into FMA, which GCC and Clang otherwise do differently under -march=native.
# Keeps the scalar reference kernel and all results bit-identical across
# compilers. Explicit intrinsics (e.g. the AVX2 kernel) are unaffected.
add_compile_options(-ffp-contract=off)
if(HNSW_NATIVE)
  add_compile_options(-march=native)
endif()

if(NOT HNSW_SANITIZE STREQUAL "off")
  if(NOT HNSW_SANITIZE MATCHES "^(address|undefined|address,undefined)$")
    message(FATAL_ERROR
      "HNSW_SANITIZE must be off, address, undefined, or address,undefined "
      "(got '${HNSW_SANITIZE}')")
  endif()
  add_compile_options(-fsanitize=${HNSW_SANITIZE} -fno-omit-frame-pointer)
  add_link_options(-fsanitize=${HNSW_SANITIZE})
endif()

add_library(hnsw STATIC
  src/io.cpp
  src/distance.cpp
  src/brute_force.cpp
  src/hnsw.cpp
  src/search.cpp
  src/aligned_buffer.cpp)
target_include_directories(hnsw PUBLIC include)
target_compile_definitions(hnsw PUBLIC
  HNSW_OPT_EPOCH_VISITED=$<BOOL:${HNSW_OPT_EPOCH_VISITED}>
  HNSW_OPT_ALLOC_FREE_HEAPS=$<BOOL:${HNSW_OPT_ALLOC_FREE_HEAPS}>
  HNSW_OPT_PREFETCH=$<BOOL:${HNSW_OPT_PREFETCH}>)

enable_testing()

function(hnsw_add_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE hnsw)
  target_include_directories(${name} PRIVATE tests)
  add_test(NAME ${name} COMMAND ${name})
  # Exit code 77 means "skipped" (e.g. feature disabled or unsupported CPU).
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

hnsw_add_test(test_distance)
hnsw_add_test(test_io)
hnsw_add_test(test_brute_force)
hnsw_add_test(test_recall)
hnsw_add_test(test_hnsw_invariants)
hnsw_add_test(test_hnsw_recall)
hnsw_add_test(test_hnsw_connectivity)
hnsw_add_test(test_hnsw_heuristic)
hnsw_add_test(test_hnsw_edge_cases)
hnsw_add_test(test_hnsw_determinism)
hnsw_add_test(test_layout)
hnsw_add_test(test_scratch)
hnsw_add_test(test_alloc_free)
hnsw_add_test(test_golden)
