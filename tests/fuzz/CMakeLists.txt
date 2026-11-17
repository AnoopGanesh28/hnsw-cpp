# Fuzz targets (see README.md here). The replay drivers and the seed generator
# are built in every configuration; the libFuzzer binaries only with
# HNSW_BUILD_FUZZ=ON.

set(_fuzz_dir ${CMAKE_CURRENT_SOURCE_DIR})

foreach(_target loader fvecs)
  add_executable(replay_${_target} fuzz_${_target}.cpp replay_main.cpp)
  target_link_libraries(replay_${_target} PRIVATE hnsw)
  target_include_directories(replay_${_target} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
  add_test(NAME fuzz_replay_${_target}
           COMMAND replay_${_target} ${_fuzz_dir}/seeds/${_target} ${_fuzz_dir}/corpus/${_target}
                   ${_fuzz_dir}/crashes/${_target})
endforeach()

add_executable(make_seeds make_seeds.cpp)
target_link_libraries(make_seeds PRIVATE hnsw)
target_include_directories(make_seeds PRIVATE ${PROJECT_SOURCE_DIR}/tests)

if(HNSW_BUILD_FUZZ)
  add_executable(fuzz_loader fuzz_loader.cpp fuzz_loader_mutator.cpp)
  add_executable(fuzz_fvecs fuzz_fvecs.cpp)
  foreach(_target fuzz_loader fuzz_fvecs)
    target_link_libraries(${_target} PRIVATE hnsw)
    target_include_directories(${_target} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
    target_link_options(${_target} PRIVATE -fsanitize=fuzzer)
  endforeach()
endif()
