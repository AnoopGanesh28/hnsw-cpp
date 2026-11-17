# Python extension hnsw_cpp._core and the hnsw_cpp package around it.
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
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
