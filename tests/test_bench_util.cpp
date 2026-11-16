// Unit tests for the benchmark utilities: VmHWM reader, JSON writer/parser,
// and the median/min/max helper.
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "json.h"
#include "proc.h"
#include "stats.h"
#include "test_helper.h"

namespace {

using bench::json::Value;

void write_text(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::trunc);
  out << text;
}

void test_status_reader() {
  test::TempFile f("status");
  write_text(f.path(),
             "Name:\tbench\nVmPeak:\t  999999 kB\nVmHWM:\t   12345 kB\nVmRSS:\t    2048 kB\n"
             "Threads:\t1\n");
  CHECK_EQ(bench::read_status_kib(f.path(), "VmHWM"), 12345);
  CHECK_EQ(bench::read_status_kib(f.path(), "VmRSS"), 2048);
  // Prefix keys must not match a longer key ("Vm" is not "VmPeak").
  CHECK_THROWS(bench::read_status_kib(f.path(), "Vm"), std::runtime_error, "no Vm line");
  CHECK_THROWS(bench::read_status_kib(f.path(), "VmSwap"), std::runtime_error, "no VmSwap");
  // A line without the kB unit is not a memory value.
  CHECK_THROWS(bench::read_status_kib(f.path(), "Threads"), std::runtime_error, "malformed");

  test::TempFile bad("status_bad");
  write_text(bad.path(), "VmHWM:\t abc kB\n");
  CHECK_THROWS(bench::read_status_kib(bad.path(), "VmHWM"), std::runtime_error, "malformed");
  write_text(bad.path(), "VmHWM:\t -5 kB\n");
  CHECK_THROWS(bench::read_status_kib(bad.path(), "VmHWM"), std::runtime_error, "malformed");
  write_text(bad.path(), "VmHWM:\t 5 MB\n");
  CHECK_THROWS(bench::read_status_kib(bad.path(), "VmHWM"), std::runtime_error, "malformed");
  CHECK_THROWS(bench::read_status_kib("/nonexistent/status", "VmHWM"), std::runtime_error,
               "cannot read");
}

void test_vm_hwm_tracks_peak() {
  const int64_t before = bench::vm_hwm_kib();
  CHECK(before > 0);
  // Touch every page of 64 MiB through a volatile pointer (so the compiler
  // cannot drop the allocation); the high-water mark must rise by most of it.
  const size_t bytes = size_t{64} << 20;
  std::vector<char> block(bytes);
  volatile char* p = block.data();
  for (size_t i = 0; i < bytes; i += 4096) p[i] = 1;
  const int64_t after = bench::vm_hwm_kib();
  CHECK(after >= before + 60 * 1024);
  // The mark is a peak and stays up when the memory is released, up to the
  // kernel's batching of per-CPU RSS counters (observed: a few hundred kB).
  block = std::vector<char>();
  CHECK(bench::vm_hwm_kib() >= after - 1024);
}

void test_json_dump() {
  Value doc = Value::object();
  doc.set("int", 42);
  doc.set("neg", -7L);
  doc.set("size", size_t{1} << 40);
  doc.set("double", 0.5);
  doc.set("whole_double", 3.0);
  doc.set("bool", true);
  doc.set("null", Value());
  doc.set("text", "a\"b\\c\n\t\x01");
  doc.set("empty_array", Value::array());
  doc.set("empty_object", Value::object());
  doc.set("list", Value::array_of(std::vector<size_t>{1, 2, 3}));
  const std::string s = bench::json::dump(doc);
  CHECK(s.find("\"int\": 42") != std::string::npos);
  CHECK(s.find("\"neg\": -7") != std::string::npos);
  CHECK(s.find("\"size\": 1099511627776") != std::string::npos);
  CHECK(s.find("\"double\": 0.5") != std::string::npos);
  CHECK(s.find("\"whole_double\": 3.0") != std::string::npos);
  CHECK(s.find("\"null\": null") != std::string::npos);
  CHECK(s.find("\"text\": \"a\\\"b\\\\c\\n\\t\\u0001\"") != std::string::npos);
  CHECK(s.find("\"empty_array\": []") != std::string::npos);
  CHECK(s.find("\"empty_object\": {}") != std::string::npos);
  // Keys come out in insertion order.
  CHECK(s.find("\"int\"") < s.find("\"neg\"") && s.find("\"neg\"") < s.find("\"list\""));
  // set() on an existing key replaces in place.
  doc.set("int", 43);
  CHECK_EQ(doc.size(), size_t{11});
  CHECK_EQ(doc.at("int").as_int(), 43);

  CHECK_THROWS(Value(std::numeric_limits<double>::quiet_NaN()), bench::json::Error, "non-finite");
  CHECK_THROWS(Value(std::numeric_limits<double>::infinity()), bench::json::Error, "non-finite");
  CHECK_THROWS(Value(std::numeric_limits<uint64_t>::max()), bench::json::Error, "exceeds int64");
}

void test_json_round_trip() {
  Value doc = Value::object();
  const std::vector<double> doubles = {0.1, 1.0 / 3.0, 1e-300, 6.02214076e23, -2.5,
                                       std::numeric_limits<double>::min(),
                                       std::numeric_limits<double>::max()};
  doc.set("doubles", Value::array_of(doubles));
  doc.set("max_int", std::numeric_limits<int64_t>::max());
  doc.set("min_int", std::numeric_limits<int64_t>::min());
  doc.set("utf8", "caf\xc3\xa9 \xe2\x88\xa9");
  Value nested = Value::object();
  nested.set("a", Value::array().push(Value::object().set("b", false)));
  doc.set("nested", nested);

  const Value back = bench::json::parse(bench::json::dump(doc));
  const Value& d = back.at("doubles");
  CHECK_EQ(d.size(), doubles.size());
  for (size_t i = 0; i < doubles.size(); ++i) {
    CHECK(d.at(i).type() == Value::Type::Double);
    CHECK(d.at(i).as_double() == doubles[i]);  // exact, not approximate
  }
  CHECK_EQ(back.at("max_int").as_int(), std::numeric_limits<int64_t>::max());
  CHECK_EQ(back.at("min_int").as_int(), std::numeric_limits<int64_t>::min());
  CHECK_EQ(back.at("utf8").as_string(), std::string("caf\xc3\xa9 \xe2\x88\xa9"));
  CHECK_EQ(back.at("nested").at("a").at(0).at("b").as_bool(), false);

  test::TempFile f("json");
  bench::json::write_file(f.path(), doc);
  CHECK_EQ(bench::json::dump(bench::json::read_file(f.path())), bench::json::dump(doc));
}

void test_json_parse() {
  const Value v = bench::json::parse(
      " {\"a\": [1, -0, 2.5e3, -1E-2, true, null], \"s\": \"\\u00e9\\ud83d\\ude00\\/\"} ");
  CHECK_EQ(v.at("a").at(0).as_int(), 1);
  CHECK(v.at("a").at(1).type() == Value::Type::Int);
  CHECK_EQ(v.at("a").at(2).as_double(), 2500.0);
  CHECK_EQ(v.at("a").at(3).as_double(), -0.01);
  CHECK(v.at("a").at(5).is_null());
  CHECK_EQ(v.at("s").as_string(), std::string("\xc3\xa9\xf0\x9f\x98\x80/"));
  // Integers beyond int64 fall back to double.
  CHECK(bench::json::parse("99999999999999999999").type() == Value::Type::Double);

  const char* bad[] = {"",          "{",         "[1,]",        "{\"a\" 1}",  "{\"a\":1,}",
                       "01",        "1.",        "1e",          "-",          "\"abc",
                       "\"\\x\"",   "\"\\ud800\"", "\"\x01\"",  "tru",        "[1] 2",
                       "{\"a\":1,\"a\":2}",       "nan",        "1e999"};
  for (const char* text : bad) {
    CHECK_THROWS(bench::json::parse(text), bench::json::Error, "json parse error");
  }
  CHECK_THROWS(bench::json::parse(std::string(300, '[') + std::string(300, ']')),
               bench::json::Error, "too deep");

  CHECK_THROWS(v.at("missing"), bench::json::Error, "missing key");
  CHECK_THROWS(v.at("a").at(99), bench::json::Error, "out of range");
  CHECK_THROWS(v.at("s").as_int(), bench::json::Error, "as_int on a string");
}

void test_summarize() {
  const bench::Summary odd = bench::summarize({5.0, 1.0, 3.0});
  CHECK_EQ(odd.median, 3.0);
  CHECK_EQ(odd.min, 1.0);
  CHECK_EQ(odd.max, 5.0);
  const bench::Summary even = bench::summarize({4.0, 1.0, 3.0, 2.0});
  CHECK_EQ(even.median, 2.5);
  CHECK_EQ(even.min, 1.0);
  CHECK_EQ(even.max, 4.0);
  const bench::Summary one = bench::summarize({7.25});
  CHECK_EQ(one.median, 7.25);
  CHECK_EQ(one.min, 7.25);
  CHECK_EQ(one.max, 7.25);
  const bench::Summary ties = bench::summarize({2.0, 2.0, 9.0, 2.0});
  CHECK_EQ(ties.median, 2.0);
  CHECK_THROWS(bench::summarize({}), std::invalid_argument, "no samples");

  // samples_json keeps the raw samples in their original order.
  const Value j = bench::samples_json({3.0, 1.0, 2.0});
  CHECK_EQ(j.at("raw").size(), size_t{3});
  CHECK_EQ(j.at("raw").at(0).as_double(), 3.0);
  CHECK_EQ(j.at("raw").at(1).as_double(), 1.0);
  CHECK_EQ(j.at("median").as_double(), 2.0);
  CHECK_EQ(j.at("min").as_double(), 1.0);
  CHECK_EQ(j.at("max").as_double(), 3.0);
}

}  // namespace

int main() {
  test_status_reader();
  test_vm_hwm_tracks_peak();
  test_json_dump();
  test_json_round_trip();
  test_json_parse();
  test_summarize();
  return test_finish();
}
