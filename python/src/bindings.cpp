// pybind11 bindings for the HNSW library: hnsw_cpp.Index (build, search, save),
// hnsw_cpp.LoadedIndex (read-only, from a saved file) and hnsw_cpp.load.
//
// Threading contract: any number of searches may run at once; add() is
// exclusive. Each index wrapper owns a std::shared_mutex that is taken only
// after the GIL is released, so a thread waiting for the lock never holds the
// GIL, and a thread holding the lock never needs the GIL to finish.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw/index_file.h"
#include "hnsw/thread_pool.h"

namespace py = pybind11;

namespace {

constexpr size_t kMaxThreads = 1024;

// A validated float32 matrix argument. `array` keeps the buffer alive for the
// whole call; `data` points at row 0.
struct Matrix {
  py::array array;
  const float* data;
  size_t rows;
};

std::string type_name(const py::handle& obj) {
  return py::cast<std::string>(py::type::handle_of(obj).attr("__name__"));
}

// Accepts exactly a C-contiguous, aligned, native-endian float32 numpy array
// of shape (dim,) (one row) or (rows, dim). Never copies or converts.
Matrix as_matrix(const py::handle& obj, const char* name, size_t dim) {
  const std::string who = name;
  const std::string fix = "; convert with np.ascontiguousarray(x, dtype=np.float32)";
  if (!py::isinstance<py::array>(obj)) {
    throw py::type_error(who + " must be a numpy.ndarray of dtype float32, got " +
                         type_name(obj) + fix);
  }
  auto array = py::reinterpret_borrow<py::array>(obj);
  if (!py::array_t<float>::check_(array)) {
    throw py::type_error(who + " must have dtype float32, got " +
                         py::cast<std::string>(py::str(array.dtype())) + fix);
  }
  if (!(array.flags() & py::array::c_style)) {
    throw py::value_error(who + " must be C-contiguous" + fix);
  }
  if (!(array.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_)) {
    throw py::value_error(who + " must be aligned" + fix);
  }
  const auto ndim = static_cast<size_t>(array.ndim());
  if (ndim != 1 && ndim != 2) {
    throw py::value_error(who + " must be 1-D (one vector) or 2-D (one vector per row), got " +
                          std::to_string(ndim) + "-D");
  }
  const auto last = static_cast<size_t>(array.shape(array.ndim() - 1));
  if (last != dim) {
    throw py::value_error(who + " has vectors of length " + std::to_string(last) +
                          " but the index has dim " + std::to_string(dim));
  }
  const size_t rows = ndim == 1 ? 1 : static_cast<size_t>(array.shape(0));
  return {array, static_cast<const float*>(array.data()), rows};
}

struct SearchParams {
  size_t k, ef, threads;
};

SearchParams check_search_params(int64_t k, int64_t ef, int64_t threads) {
  if (k < 1) throw py::value_error("k must be >= 1 (got " + std::to_string(k) + ")");
  if (ef < 1) throw py::value_error("ef must be >= 1 (got " + std::to_string(ef) + ")");
  if (threads < 0) {
    throw py::value_error("threads must be >= 0 (0 = all hardware threads, got " +
                          std::to_string(threads) + ")");
  }
  if (threads > static_cast<int64_t>(kMaxThreads)) {
    throw py::value_error("threads must be at most " + std::to_string(kMaxThreads) + " (got " +
                          std::to_string(threads) + ")");
  }
  return {static_cast<size_t>(k), static_cast<size_t>(ef), static_cast<size_t>(threads)};
}

// The thread pool of one index wrapper, kept between calls and replaced only
// when a call asks for a different size. A call holds a shared_ptr, so a
// replaced pool lives until its last user is done.
class PoolCache {
 public:
  std::shared_ptr<hnsw::ThreadPool> get(size_t threads) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pool_ || requested_ != threads) {
      pool_ = std::make_shared<hnsw::ThreadPool>(threads);
      requested_ = threads;
    }
    return pool_;
  }

 private:
  std::mutex mutex_;
  std::shared_ptr<hnsw::ThreadPool> pool_;
  size_t requested_ = 0;
};

// Searches nq queries into ids (nq * k, kNoNeighbor padded) and dists
// (+infinity padded). threads == 1 runs in the calling thread with a
// thread_local scratch, so concurrent callers never contend; any other value
// goes through the cached pool (concurrent calls on one pool take turns).
template <typename Index>
void run_queries(const Index& index, const float* queries, size_t nq, const SearchParams& p,
                 PoolCache& pools, uint32_t* ids, float* dists) {
  if (p.threads == 1) {
    thread_local hnsw::SearchScratch scratch;
    for (size_t q = 0; q < nq; ++q) {
      uint32_t* row_ids = ids + q * p.k;
      float* row_dists = dists + q * p.k;
      const size_t got = index.search_into(queries + q * index.dim(), p.k, p.ef, scratch,
                                           row_ids, row_dists);
      std::fill(row_ids + got, row_ids + p.k, hnsw::kNoNeighbor);
      std::fill(row_dists + got, row_dists + p.k, std::numeric_limits<float>::infinity());
    }
  } else {
    index.search_batch(queries, nq, p.k, p.ef, ids, dists, *pools.get(p.threads));
  }
}

// Validates arguments, then runs the search with the GIL released and returns
// (ids int64 [nq, k], dists float32 [nq, k]). `with_index(fn)` takes the
// shared lock, checks the index is open and calls fn(index).
template <typename WithIndex>
py::tuple search_arrays(size_t dim, const py::handle& queries_arg, int64_t k_arg, int64_t ef_arg,
                        int64_t threads_arg, PoolCache& pools, WithIndex&& with_index) {
  const SearchParams p = check_search_params(k_arg, ef_arg, threads_arg);
  const Matrix queries = as_matrix(queries_arg, "queries", dim);
  size_t cells = 0;
  if (__builtin_mul_overflow(queries.rows, p.k, &cells)) {
    throw py::value_error("queries x k is too large");
  }
  const auto shape = std::vector<py::ssize_t>{static_cast<py::ssize_t>(queries.rows),
                                              static_cast<py::ssize_t>(p.k)};
  py::array_t<int64_t> ids(shape);
  py::array_t<float> dists(shape);
  int64_t* out_ids = ids.mutable_data();
  float* out_dists = dists.mutable_data();
  {
    py::gil_scoped_release release_gil;
    with_index([&](const auto& index) {
      std::vector<uint32_t> raw(cells);
      run_queries(index, queries.data, queries.rows, p, pools, raw.data(), out_dists);
      for (size_t i = 0; i < cells; ++i) {
        out_ids[i] = raw[i] == hnsw::kNoNeighbor ? -1 : static_cast<int64_t>(raw[i]);
      }
    });
  }
  return py::make_tuple(ids, dists);
}

constexpr const char* kSearchDoc = R"doc(Approximate k-nearest-neighbor search.

Parameters
----------
queries : numpy.ndarray
    float32, C-contiguous, shape (nq, dim), or shape (dim,) for one query.
k : int
    Number of neighbors per query, >= 1.
ef : int
    Candidate list size, >= 1 (larger is slower and more accurate; at least k
    candidates are always used).
threads : int, default 1
    1 searches in the calling thread, 0 uses all hardware threads, any other
    value uses a pool of that size (at most 1024). The pool is kept between
    calls. Concurrent calls with threads != 1 on one index take turns on its
    pool; threads=1 calls run fully in parallel.

Returns
-------
(ids, dists) : (numpy.ndarray int64 [nq, k], numpy.ndarray float32 [nq, k])
    Each row is ascending by distance. Distances are SQUARED L2 (no square
    root). Missing results (k larger than the number of reachable vectors) have
    id -1 and distance +inf.

The GIL is released while searching.)doc";

// ---------------------------------------------------------------------------

class PyIndex {
 public:
  PyIndex(size_t dim, const hnsw::HnswParams& params) : index_(dim, params) {}

  py::array_t<int64_t> add(const py::handle& vectors) {
    const Matrix m = as_matrix(vectors, "vectors", index_.dim());
    py::array_t<int64_t> ids(static_cast<py::ssize_t>(m.rows));
    int64_t* out = ids.mutable_data();
    {
      py::gil_scoped_release release_gil;
      // Reject the whole batch before adding any row of it.
      for (size_t r = 0; r < m.rows; ++r) {
        for (size_t i = 0; i < index_.dim(); ++i) {
          if (!std::isfinite(m.data[r * index_.dim() + i])) {
            throw py::value_error("vectors[" + std::to_string(r) + "] element " +
                                  std::to_string(i) +
                                  " is NaN or infinite; nothing was added");
          }
        }
      }
      std::unique_lock<std::shared_mutex> lock(mutex_);
      if (m.rows > std::numeric_limits<uint32_t>::max() - index_.size()) {
        throw py::value_error("adding " + std::to_string(m.rows) +
                              " vectors would exceed the maximum index size");
      }
      for (size_t r = 0; r < m.rows; ++r) out[r] = index_.add(m.data + r * index_.dim());
    }
    return ids;
  }

  py::tuple search(const py::handle& queries, int64_t k, int64_t ef, int64_t threads) {
    return search_arrays(index_.dim(), queries, k, ef, threads, pools_, [&](auto&& fn) {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      fn(index_);
    });
  }

  void save(const std::filesystem::path& path) const {
    py::gil_scoped_release release_gil;
    std::shared_lock<std::shared_mutex> lock(mutex_);
    index_.save(path.string());
  }

  size_t size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return index_.size();
  }
  size_t dim() const { return index_.dim(); }
  size_t M() const { return index_.params().M; }
  size_t ef_construction() const { return index_.params().ef_construction; }
  uint64_t seed() const { return index_.params().seed; }

 private:
  hnsw::HnswIndex index_;
  mutable std::shared_mutex mutex_;
  PoolCache pools_;
};

class PyLoaded {
 public:
  explicit PyLoaded(hnsw::LoadedIndex index) : dim_(index.dim()), index_(std::move(index)) {}

  py::tuple search(const py::handle& queries, int64_t k, int64_t ef, int64_t threads) {
    return search_arrays(dim_, queries, k, ef, threads, pools_, [&](auto&& fn) {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      require_open();
      fn(*index_);
    });
  }

  size_t size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    require_open();
    return index_->size();
  }
  size_t dim() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    require_open();
    return dim_;
  }

  bool closed() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return !index_.has_value();
  }

  // Waits for running searches, then releases the mapping. Idempotent.
  void close() {
    py::gil_scoped_release release_gil;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    index_.reset();
  }

 private:
  void require_open() const {
    if (!index_.has_value()) throw py::value_error("LoadedIndex is closed");
  }

  size_t dim_;
  std::optional<hnsw::LoadedIndex> index_;
  mutable std::shared_mutex mutex_;
  PoolCache pools_;
};

// The loader's single error type covers both unreadable and invalid files.
// Files that cannot be opened at all are reported as OSError (with errno, so
// FileNotFoundError and PermissionError come out right); everything the
// loader rejects after that is a ValueError.
void require_openable(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    PyErr_SetFromErrnoWithFilename(PyExc_OSError, path.c_str());
    throw py::error_already_set();
  }
  ::close(fd);
}

std::unique_ptr<PyLoaded> load(const std::filesystem::path& path, const std::string& mode,
                               const std::string& validate) {
  hnsw::LoadOptions options;
  if (mode == "mmap") {
    options.mode = hnsw::LoadMode::Mmap;
  } else if (mode == "read") {
    options.mode = hnsw::LoadMode::Read;
  } else {
    throw py::value_error("mode must be 'mmap' or 'read' (got '" + mode + "')");
  }
  if (validate == "deep") {
    options.validate_deep = true;
  } else if (validate == "shallow") {
    options.validate_deep = false;
  } else {
    throw py::value_error("validate must be 'deep' or 'shallow' (got '" + validate + "')");
  }
  require_openable(path.string());
  py::gil_scoped_release release_gil;
  try {
    return std::make_unique<PyLoaded>(hnsw::LoadedIndex::open(path.string(), options));
  } catch (const hnsw::IndexFileError& e) {
    throw py::value_error(e.what());
  }
}

std::unique_ptr<PyIndex> make_index(int64_t dim, int64_t M, int64_t ef_construction,
                                    uint64_t seed) {
  using hnsw::format::kMaxDim;
  using hnsw::format::kMaxM;
  // The limits are the file format's: a larger index could never be saved.
  if (dim < 1 || dim > static_cast<int64_t>(kMaxDim)) {
    throw py::value_error("dim must be in [1, " + std::to_string(kMaxDim) + "] (got " +
                          std::to_string(dim) + ")");
  }
  if (M < 2 || M > static_cast<int64_t>(kMaxM)) {
    throw py::value_error("M must be in [2, " + std::to_string(kMaxM) + "] (got " +
                          std::to_string(M) + ")");
  }
  if (ef_construction < 1 || ef_construction > std::numeric_limits<uint32_t>::max()) {
    throw py::value_error("ef_construction must be in [1, 4294967295] (got " +
                          std::to_string(ef_construction) + ")");
  }
  hnsw::HnswParams params;
  params.M = static_cast<size_t>(M);
  params.ef_construction = static_cast<size_t>(ef_construction);
  params.seed = seed;
  return std::make_unique<PyIndex>(static_cast<size_t>(dim), params);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "HNSW approximate nearest-neighbor index (squared L2), from-scratch C++17.";

  // save() reports I/O failures with IndexFileError.
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const hnsw::IndexFileError& e) {
      PyErr_SetString(PyExc_OSError, e.what());
    }
  });

  py::class_<PyIndex>(m, "Index", R"doc(Mutable HNSW index over float32 vectors, squared L2 distance.

Ids are assigned from 0 in insertion order. Any number of searches may run
concurrently; add() is exclusive (it waits until no search is running; the
lock does not guarantee that a waiting add() goes before newly started
searches). Not safe to use across fork().

Parameters
----------
dim : int
    Vector length, 1 to 65536.
M : int, default 16
    Maximum links per node above layer 0 (layer 0 allows 2 * M), 2 to 1024.
ef_construction : int, default 200
    Candidate list size while inserting, >= 1.
seed : int, default 100
    Seed for level sampling. The same seed and insertion order give the same
    graph.)doc")
      .def(py::init(&make_index), py::arg("dim"), py::arg("M") = 16,
           py::arg("ef_construction") = 200, py::arg("seed") = 100)
      .def("add", &PyIndex::add, py::arg("vectors"),
           R"doc(Add vectors and return their ids.

Parameters
----------
vectors : numpy.ndarray
    float32, C-contiguous, shape (n, dim), or shape (dim,) for one vector.
    Nothing is copied or converted for you; the array must not be modified by
    another thread during the call.

Returns
-------
numpy.ndarray of int64, shape (n,)
    The ids assigned, consecutive and ascending.

Raises ValueError if any element is NaN or infinite; in that case nothing from
the call is added, and the index is unchanged.)doc")
      .def("search", &PyIndex::search, py::arg("queries"), py::arg("k"), py::arg("ef"),
           py::arg("threads") = 1, kSearchDoc)
      .def("save", &PyIndex::save, py::arg("path"),
           R"doc(Write the index to path (str or os.PathLike).

The file is written to a temporary file in the same directory, flushed to disk
and renamed over path, so readers see the old file or the complete new one.
Concurrent searches may continue; add() waits. Raises OSError on I/O errors.
Load the file with hnsw_cpp.load.)doc")
      .def_property_readonly("size", &PyIndex::size, "Number of vectors in the index.")
      .def_property_readonly("dim", &PyIndex::dim, "Vector length.")
      .def_property_readonly("M", &PyIndex::M, "Maximum links per node above layer 0.")
      .def_property_readonly("ef_construction", &PyIndex::ef_construction,
                             "Candidate list size used while inserting.")
      .def_property_readonly("seed", &PyIndex::seed, "Level-sampling seed.");

  py::class_<PyLoaded>(m, "LoadedIndex", R"doc(Read-only index loaded from a file by hnsw_cpp.load.

Searches are bit-identical to the Index the file was saved from. It owns its
memory mapping or buffer, so it stays valid after the source Index is deleted.
Any number of concurrent searches are allowed. close() (or deleting the
object) releases the mapping; with mode="mmap" the file must not be truncated
or modified while the index is open. Not safe to use across fork().)doc")
      .def("search", &PyLoaded::search, py::arg("queries"), py::arg("k"), py::arg("ef"),
           py::arg("threads") = 1, kSearchDoc)
      .def("close", &PyLoaded::close,
           "Release the mapping or buffer after running searches finish. Idempotent; "
           "later searches raise ValueError.")
      .def_property_readonly("closed", &PyLoaded::closed, "True after close().")
      .def_property_readonly("size", &PyLoaded::size, "Number of vectors in the index.")
      .def_property_readonly("dim", &PyLoaded::dim, "Vector length.")
      .def("__enter__", [](py::object self) { return self; },
           "Return self; the index is closed on exit.")
      .def("__exit__", [](PyLoaded& self, const py::args&) { self.close(); },
           "Close the index.");

  m.def("load", &load, py::arg("path"), py::arg("mode") = "mmap", py::arg("validate") = "deep",
        R"doc(Load an index file written by Index.save.

Parameters
----------
path : str or os.PathLike
mode : "mmap" or "read", default "mmap"
    "mmap" maps the file read-only and searches it in place; "read" reads it
    into memory.
validate : "deep" or "shallow", default "deep"
    "deep" checks the whole graph and every vector on load, which makes
    searching any file that loads memory-safe, whatever its bytes.
    "shallow" checks only the header and section bounds; it requires a
    TRUSTED file, because a corrupt or malicious file can then make searches
    read out of bounds.

Raises OSError if the file cannot be opened, ValueError if the file is not a
valid index or an argument is invalid.)doc");
}
