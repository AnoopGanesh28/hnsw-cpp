#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bench::json {

// Thrown by parse() on malformed input and by accessors on a type mismatch.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// A JSON value. Objects keep insertion order (the writer emits keys in the
// order they were set). Integers and doubles are kept apart so that counts
// round-trip exactly; as_double() accepts both.
class Value {
 public:
  enum class Type { Null, Bool, Int, Double, String, Array, Object };

  Value() = default;
  Value(std::nullptr_t) {}
  Value(bool b) : type_(Type::Bool), bool_(b) {}
  Value(int v) : type_(Type::Int), int_(v) {}
  Value(long v) : type_(Type::Int), int_(v) {}
  Value(long long v) : type_(Type::Int), int_(v) {}
  Value(unsigned v) : type_(Type::Int), int_(v) {}
  // Throws Error if v exceeds INT64_MAX.
  Value(unsigned long v);
  Value(unsigned long long v);
  // Throws Error for NaN or infinity (not representable in JSON).
  Value(double v);
  Value(const char* s) : type_(Type::String), string_(s) {}
  Value(std::string s) : type_(Type::String), string_(std::move(s)) {}

  static Value array() { Value v; v.type_ = Type::Array; return v; }
  static Value object() { Value v; v.type_ = Type::Object; return v; }
  template <typename T>
  static Value array_of(const std::vector<T>& items) {
    Value v = array();
    for (const T& x : items) v.push(Value(x));
    return v;
  }

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }

  bool as_bool() const;
  int64_t as_int() const;
  double as_double() const;  // Int or Double
  const std::string& as_string() const;
  const std::vector<Value>& items() const;  // Array
  const std::vector<std::pair<std::string, Value>>& members() const;  // Object

  // Array: appends. Returns *this.
  Value& push(Value v);
  // Object: sets key (replacing an existing value). Returns *this.
  Value& set(const std::string& key, Value v);
  // Object: the value for key, or nullptr.
  const Value* find(const std::string& key) const;
  // Object: the value for key; throws Error if absent.
  const Value& at(const std::string& key) const;
  // Array: element i; throws Error if out of range.
  const Value& at(size_t i) const;
  // Array or object element count.
  size_t size() const;

 private:
  void expect(Type t, const char* what) const;

  Type type_ = Type::Null;
  bool bool_ = false;
  int64_t int_ = 0;
  double double_ = 0.0;
  std::string string_;
  std::vector<Value> array_;
  std::vector<std::pair<std::string, Value>> object_;
};

// Serializes v with two-space indentation. Doubles are written with 17
// significant digits, so parse(dump(v)) reproduces every double exactly.
std::string dump(const Value& v);

// Parses one JSON document (RFC 8259); surrounding whitespace is allowed,
// anything else after the value is an error. Numbers without a fraction or
// exponent that fit in int64 become Int, all others Double.
Value parse(const std::string& text);

// Writes dump(v) plus a newline to path via a temporary file and rename, so a
// crash never leaves a truncated result file. Throws std::runtime_error.
void write_file(const std::string& path, const Value& v);
// Reads and parses a file. Throws std::runtime_error or Error.
Value read_file(const std::string& path);

}  // namespace bench::json
