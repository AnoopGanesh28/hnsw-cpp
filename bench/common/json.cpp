#include "json.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

namespace bench::json {

namespace {

const char* type_name(Value::Type t) {
  switch (t) {
    case Value::Type::Null: return "null";
    case Value::Type::Bool: return "bool";
    case Value::Type::Int: return "int";
    case Value::Type::Double: return "double";
    case Value::Type::String: return "string";
    case Value::Type::Array: return "array";
    case Value::Type::Object: return "object";
  }
  return "?";
}

int64_t checked_int(unsigned long long v) {
  if (v > static_cast<unsigned long long>(std::numeric_limits<int64_t>::max())) {
    throw Error("json: integer " + std::to_string(v) + " exceeds int64");
  }
  return static_cast<int64_t>(v);
}

void dump_string(const std::string& s, std::string& out) {
  out += '"';
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += ch;
        }
    }
  }
  out += '"';
}

void dump_value(const Value& v, int depth, std::string& out) {
  const std::string pad(static_cast<size_t>(depth + 1) * 2, ' ');
  const std::string close_pad(static_cast<size_t>(depth) * 2, ' ');
  switch (v.type()) {
    case Value::Type::Null: out += "null"; return;
    case Value::Type::Bool: out += v.as_bool() ? "true" : "false"; return;
    case Value::Type::Int: out += std::to_string(v.as_int()); return;
    case Value::Type::Double: {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.17g", v.as_double());
      std::string s = buf;
      // Keep doubles recognizable as doubles after a round trip.
      if (s.find_first_of(".eE") == std::string::npos) s += ".0";
      out += s;
      return;
    }
    case Value::Type::String: dump_string(v.as_string(), out); return;
    case Value::Type::Array: {
      if (v.size() == 0) { out += "[]"; return; }
      out += "[\n";
      for (size_t i = 0; i < v.size(); ++i) {
        out += pad;
        dump_value(v.items()[i], depth + 1, out);
        out += i + 1 < v.size() ? ",\n" : "\n";
      }
      out += close_pad + "]";
      return;
    }
    case Value::Type::Object: {
      if (v.size() == 0) { out += "{}"; return; }
      out += "{\n";
      for (size_t i = 0; i < v.size(); ++i) {
        out += pad;
        dump_string(v.members()[i].first, out);
        out += ": ";
        dump_value(v.members()[i].second, depth + 1, out);
        out += i + 1 < v.size() ? ",\n" : "\n";
      }
      out += close_pad + "}";
      return;
    }
  }
}

class Parser {
 public:
  explicit Parser(const std::string& text) : s_(text) {}

  Value document() {
    skip_ws();
    Value v = value(0);
    skip_ws();
    if (pos_ != s_.size()) fail("unexpected trailing characters");
    return v;
  }

 private:
  static constexpr int kMaxDepth = 256;

  [[noreturn]] void fail(const std::string& msg) const {
    throw Error("json parse error at offset " + std::to_string(pos_) + ": " + msg);
  }

  void skip_ws() {
    while (pos_ < s_.size() &&
           (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) {
      ++pos_;
    }
  }

  char peek() const {
    if (pos_ >= s_.size()) fail("unexpected end of input");
    return s_[pos_];
  }

  void expect_literal(const char* lit) {
    for (const char* p = lit; *p; ++p) {
      if (pos_ >= s_.size() || s_[pos_] != *p) fail(std::string("expected '") + lit + "'");
      ++pos_;
    }
  }

  Value value(int depth) {
    if (depth > kMaxDepth) fail("nesting too deep");
    const char c = peek();
    switch (c) {
      case '{': return object(depth);
      case '[': return array(depth);
      case '"': return Value(string());
      case 't': expect_literal("true"); return Value(true);
      case 'f': expect_literal("false"); return Value(false);
      case 'n': expect_literal("null"); return Value();
      default:
        if (c == '-' || (c >= '0' && c <= '9')) return number();
        fail(std::string("unexpected character '") + c + "'");
    }
  }

  Value object(int depth) {
    Value obj = Value::object();
    ++pos_;  // '{'
    skip_ws();
    if (peek() == '}') { ++pos_; return obj; }
    for (;;) {
      skip_ws();
      if (peek() != '"') fail("expected object key");
      std::string key = string();
      if (obj.find(key)) fail("duplicate key '" + key + "'");
      skip_ws();
      if (peek() != ':') fail("expected ':'");
      ++pos_;
      skip_ws();
      obj.set(key, value(depth + 1));
      skip_ws();
      const char c = peek();
      ++pos_;
      if (c == '}') return obj;
      if (c != ',') fail("expected ',' or '}'");
    }
  }

  Value array(int depth) {
    Value arr = Value::array();
    ++pos_;  // '['
    skip_ws();
    if (peek() == ']') { ++pos_; return arr; }
    for (;;) {
      skip_ws();
      arr.push(value(depth + 1));
      skip_ws();
      const char c = peek();
      ++pos_;
      if (c == ']') return arr;
      if (c != ',') fail("expected ',' or ']'");
    }
  }

  unsigned hex4() {
    if (s_.size() - pos_ < 4) fail("truncated \\u escape");
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = s_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
      else fail("invalid \\u escape");
    }
    return v;
  }

  static void append_utf8(unsigned cp, std::string& out) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  std::string string() {
    ++pos_;  // opening quote
    std::string out;
    for (;;) {
      if (pos_ >= s_.size()) fail("unterminated string");
      const char c = s_[pos_++];
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
      if (c != '\\') { out += c; continue; }
      if (pos_ >= s_.size()) fail("unterminated escape");
      const char e = s_[pos_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          unsigned cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (s_.compare(pos_, 2, "\\u") != 0) fail("unpaired high surrogate");
            pos_ += 2;
            const unsigned lo = hex4();
            if (lo < 0xDC00 || lo > 0xDFFF) fail("invalid low surrogate");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            fail("unpaired low surrogate");
          }
          append_utf8(cp, out);
          break;
        }
        default: fail(std::string("invalid escape '\\") + e + "'");
      }
    }
  }

  Value number() {
    const size_t start = pos_;
    bool integral = true;
    if (s_[pos_] == '-') ++pos_;
    if (pos_ >= s_.size()) fail("truncated number");
    if (s_[pos_] == '0') {
      ++pos_;
    } else if (s_[pos_] >= '1' && s_[pos_] <= '9') {
      while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
    } else {
      fail("invalid number");
    }
    if (pos_ < s_.size() && s_[pos_] == '.') {
      integral = false;
      ++pos_;
      const size_t digits = pos_;
      while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
      if (pos_ == digits) fail("digit expected after '.'");
    }
    if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
      integral = false;
      ++pos_;
      if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
      const size_t digits = pos_;
      while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
      if (pos_ == digits) fail("digit expected in exponent");
    }
    const std::string tok = s_.substr(start, pos_ - start);
    if (integral) {
      errno = 0;
      char* end = nullptr;
      const long long v = std::strtoll(tok.c_str(), &end, 10);
      if (errno == 0 && end == tok.c_str() + tok.size()) return Value(v);
    }
    errno = 0;
    char* end = nullptr;
    const double d = std::strtod(tok.c_str(), &end);
    if (end != tok.c_str() + tok.size() || !std::isfinite(d)) fail("number out of range");
    return Value(d);
  }

  const std::string& s_;
  size_t pos_ = 0;
};

}  // namespace

Value::Value(unsigned long v) : type_(Type::Int), int_(checked_int(v)) {}
Value::Value(unsigned long long v) : type_(Type::Int), int_(checked_int(v)) {}

Value::Value(double v) : type_(Type::Double), double_(v) {
  if (!std::isfinite(v)) throw Error("json: non-finite double");
}

void Value::expect(Type t, const char* what) const {
  if (type_ != t) {
    throw Error(std::string("json: ") + what + " on a " + type_name(type_) + " value");
  }
}

bool Value::as_bool() const { expect(Type::Bool, "as_bool"); return bool_; }
int64_t Value::as_int() const { expect(Type::Int, "as_int"); return int_; }
double Value::as_double() const {
  if (type_ == Type::Int) return static_cast<double>(int_);
  expect(Type::Double, "as_double");
  return double_;
}
const std::string& Value::as_string() const { expect(Type::String, "as_string"); return string_; }
const std::vector<Value>& Value::items() const { expect(Type::Array, "items"); return array_; }
const std::vector<std::pair<std::string, Value>>& Value::members() const {
  expect(Type::Object, "members");
  return object_;
}

Value& Value::push(Value v) {
  expect(Type::Array, "push");
  array_.push_back(std::move(v));
  return *this;
}

Value& Value::set(const std::string& key, Value v) {
  expect(Type::Object, "set");
  for (auto& [k, existing] : object_) {
    if (k == key) { existing = std::move(v); return *this; }
  }
  object_.emplace_back(key, std::move(v));
  return *this;
}

const Value* Value::find(const std::string& key) const {
  expect(Type::Object, "find");
  for (const auto& [k, v] : object_) {
    if (k == key) return &v;
  }
  return nullptr;
}

const Value& Value::at(const std::string& key) const {
  const Value* v = find(key);
  if (!v) throw Error("json: missing key '" + key + "'");
  return *v;
}

const Value& Value::at(size_t i) const {
  expect(Type::Array, "at");
  if (i >= array_.size()) throw Error("json: index " + std::to_string(i) + " out of range");
  return array_[i];
}

size_t Value::size() const {
  if (type_ == Type::Array) return array_.size();
  expect(Type::Object, "size");
  return object_.size();
}

std::string dump(const Value& v) {
  std::string out;
  dump_value(v, 0, out);
  return out;
}

Value parse(const std::string& text) { return Parser(text).document(); }

void write_file(const std::string& path, const Value& v) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write '" + tmp + "'");
    out << dump(v) << '\n';
    out.flush();
    if (!out) throw std::runtime_error("write error on '" + tmp + "'");
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
    throw std::runtime_error("cannot rename '" + tmp + "' to '" + path + "'");
  }
}

Value read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open '" + path + "'");
  std::ostringstream ss;
  ss << in.rdbuf();
  return parse(ss.str());
}

}  // namespace bench::json
