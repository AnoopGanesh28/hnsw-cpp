#include "hnswlib_index.h"

#include <hnswlib/hnswlib.h>

#include <stdexcept>
#include <utility>

namespace bench {

struct HnswlibIndex::Impl {
  explicit Impl(size_t dim) : space(dim) {}
  hnswlib::L2Space space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg;
};

HnswlibIndex::HnswlibIndex(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

HnswlibIndex::HnswlibIndex(size_t dim, size_t max_elements, size_t M, size_t ef_construction,
                           size_t seed)
    : impl_(std::make_unique<Impl>(dim)) {
  impl_->alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      &impl_->space, max_elements, M, ef_construction, seed);
}

std::unique_ptr<HnswlibIndex> HnswlibIndex::load(const std::string& path, size_t dim) {
  auto impl = std::make_unique<Impl>(dim);
  impl->alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(&impl->space, path);
  return std::unique_ptr<HnswlibIndex>(new HnswlibIndex(std::move(impl)));
}

HnswlibIndex::~HnswlibIndex() = default;

void HnswlibIndex::add(const float* vec, size_t label) { impl_->alg->addPoint(vec, label); }

void HnswlibIndex::save(const std::string& path) { impl_->alg->saveIndex(path); }

void HnswlibIndex::set_ef(size_t ef) { impl_->alg->setEf(ef); }

size_t HnswlibIndex::search_into(const float* query, size_t k, uint32_t* ids,
                                 float* dists) const {
  auto heap = impl_->alg->searchKnn(query, k);
  const size_t n = heap.size();
  // The heap's top is the farthest result: fill from the back.
  for (size_t i = n; i-- > 0;) {
    ids[i] = static_cast<uint32_t>(heap.top().second);
    dists[i] = heap.top().first;
    heap.pop();
  }
  return n;
}

size_t HnswlibIndex::size() const { return impl_->alg->getCurrentElementCount(); }

json::Value hnswlib_info(size_t dim) {
  json::Value macros = json::Value::object();
#ifdef USE_SSE
  macros.set("USE_SSE", true);
#else
  macros.set("USE_SSE", false);
#endif
#ifdef USE_AVX
  macros.set("USE_AVX", true);
#else
  macros.set("USE_AVX", false);
#endif
#ifdef USE_AVX512
  macros.set("USE_AVX512", true);
#else
  macros.set("USE_AVX512", false);
#endif

  json::Value runtime = json::Value::object();
#if defined(USE_AVX) || defined(USE_SSE)
  runtime.set("AVXCapable", AVXCapable());
#endif
#if defined(USE_AVX512)
  runtime.set("AVX512Capable", AVX512Capable());
#endif

  // Constructing the space runs hnswlib's kernel selection.
  hnswlib::L2Space space(dim);
  const hnswlib::DISTFUNC<float> f = space.get_dist_func();
  std::string kernel = "unknown";
  if (f == hnswlib::L2Sqr) kernel = "L2Sqr (scalar)";
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
  if (f == hnswlib::L2SqrSIMD16ExtSSE) kernel = "L2SqrSIMD16ExtSSE";
  if (f == hnswlib::L2SqrSIMD16ExtResiduals) kernel = "L2SqrSIMD16ExtResiduals";
  if (f == hnswlib::L2SqrSIMD4Ext) kernel = "L2SqrSIMD4Ext (SSE)";
  if (f == hnswlib::L2SqrSIMD4ExtResiduals) kernel = "L2SqrSIMD4ExtResiduals";
  std::string wide = "L2SqrSIMD16ExtSSE";
#if defined(USE_AVX)
  if (f == hnswlib::L2SqrSIMD16ExtAVX) kernel = "L2SqrSIMD16ExtAVX";
  if (hnswlib::L2SqrSIMD16Ext == hnswlib::L2SqrSIMD16ExtAVX) wide = "L2SqrSIMD16ExtAVX";
#endif
#if defined(USE_AVX512)
  if (f == hnswlib::L2SqrSIMD16ExtAVX512) kernel = "L2SqrSIMD16ExtAVX512";
  if (hnswlib::L2SqrSIMD16Ext == hnswlib::L2SqrSIMD16ExtAVX512) wide = "L2SqrSIMD16ExtAVX512";
#endif
#endif

  json::Value out = json::Value::object();
  out.set("header_macros", macros);
  out.set("runtime_checks", runtime);
  out.set("dim", dim);
  out.set("l2_function", kernel);
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
  out.set("simd16_kernel", wide);
#endif
  // The AVX kernels sit behind L2SqrSIMD16Ext, which only the 16-wide entry
  // points use; the 4-wide ones are SSE only.
#if defined(USE_AVX)
  const bool uses_wide = f == hnswlib::L2SqrSIMD16Ext || f == hnswlib::L2SqrSIMD16ExtResiduals;
  out.set("avx_path_active", uses_wide && wide != "L2SqrSIMD16ExtSSE");
#else
  out.set("avx_path_active", false);
#endif
  return out;
}

}  // namespace bench
