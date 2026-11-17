# Python extension hnsw_cpp._core and the hnsw_cpp package around it.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE hnsw)

# Build-tree layout: <build>/python/hnsw_cpp/{__init__.py,_core*.so}, so that
# PYTHONPATH=<build>/python makes `import hnsw_cpp` work without installing.
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hnsw_cpp)
add_custom_command(TARGET _core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_cpp/__init__.py
          ${CMAKE_CURRENT_BINARY_DIR}/hnsw_cpp/__init__.py)

# The wheel gets the extension from here and the .py/.pyi/py.typed files from
# the package directory (pyproject.toml: wheel.packages).
install(TARGETS _core LIBRARY DESTINATION hnsw_cpp)

# Tests run against the build tree (PYTHONPATH). Skipped for sanitizer builds:
# the extension would need the sanitizer runtime preloaded into Python.
if(HNSW_SANITIZE STREQUAL "off")
  set(_hnsw_py_env "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
  add_test(NAME python_tests
           COMMAND ${Python_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  add_test(NAME python_quickstart
           COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/examples/quickstart.py)
  set_tests_properties(python_tests python_quickstart PROPERTIES
                       ENVIRONMENT "${_hnsw_py_env}" TIMEOUT 300)
else()
  message(STATUS "Python tests not registered: sanitizer build")
endif()
