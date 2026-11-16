#include "dataset.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "proc.h"

namespace bench {

namespace fs = std::filesystem;

hnsw::FvecsData read_fvecs_prefix(const std::string& path, size_t max_count) {
  const std::string who = "fvecs '" + path + "'";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + who);
  std::error_code ec;
  const uint64_t file_size = fs::file_size(path, ec);
  if (ec || file_size == 0) throw std::runtime_error(who + ": empty or unreadable");

  int32_t d = 0;
  if (!in.read(reinterpret_cast<char*>(&d), sizeof(d)) || d <= 0) {
    throw std::runtime_error(who + ": invalid dimension in record 0");
  }
  const uint64_t record_bytes = 4 + static_cast<uint64_t>(d) * sizeof(float);
  if (file_size % record_bytes != 0) {
    throw std::runtime_error(who + ": size is not a multiple of the record size");
  }
  const uint64_t in_file = file_size / record_bytes;
  if (max_count != kAllRecords && max_count > in_file) {
    throw std::runtime_error(who + " holds " + std::to_string(in_file) + " vectors, " +
                             std::to_string(max_count) + " requested");
  }
  hnsw::FvecsData out;
  out.dim = static_cast<size_t>(d);
  out.count = max_count == kAllRecords ? static_cast<size_t>(in_file) : max_count;
  out.data.resize(out.count * out.dim);
  in.seekg(0);
  for (size_t i = 0; i < out.count; ++i) {
    int32_t di = 0;
    in.read(reinterpret_cast<char*>(&di), sizeof(di));
    if (di != d) {
      throw std::runtime_error(who + ": record " + std::to_string(i) + " has dimension " +
                               std::to_string(di) + ", expected " + std::to_string(d));
    }
    in.read(reinterpret_cast<char*>(out.data.data() + i * out.dim),
            static_cast<std::streamsize>(out.dim * sizeof(float)));
    if (!in) throw std::runtime_error(who + ": read error in record " + std::to_string(i));
  }
  return out;
}

std::string sha256_file(const std::string& path) {
  const std::string out = run_command("sha256sum -b -- " + shell_quote(path));
  const std::string hex = out.substr(0, out.find(' '));
  if (hex.size() != 64 || hex.find_first_not_of("0123456789abcdef") != std::string::npos) {
    throw std::runtime_error("unexpected sha256sum output for '" + path + "'");
  }
  return hex;
}

Manifest manifest_for(const std::string& data_file) {
  const fs::path dir = fs::absolute(data_file).parent_path();
  const std::string path = (dir / "MANIFEST").string();
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("no MANIFEST next to '" + data_file +
                             "' (expected " + path +
                             "; datasets come from bench/get_sift.sh or bench_make_synthetic)");
  }
  Manifest m;
  m.dir = dir.string();
  std::string line;
  while (std::getline(in, line)) {
    const size_t colon = line.find(": ");
    if (colon == std::string::npos) continue;
    const std::string key = line.substr(0, colon);
    const std::string value = line.substr(colon + 2);
    if (key == "dataset") {
      m.dataset = value;
    } else if (key == "file") {
      std::istringstream ss(value);
      std::string name;
      ManifestFile f;
      if (!(ss >> name >> f.bytes >> f.sha256)) {
        throw std::runtime_error(path + ": malformed line '" + line + "'");
      }
      m.files[name] = f;
    } else {
      m.fields[key] = value;
    }
  }
  if (m.dataset.empty()) throw std::runtime_error(path + ": no 'dataset:' line");
  return m;
}

void write_manifest(const std::string& dir, const std::string& dataset,
                    const std::vector<std::pair<std::string, std::string>>& fields,
                    const std::vector<std::string>& file_names) {
  const std::string path = (fs::path(dir) / "MANIFEST").string();
  std::ofstream out(path, std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << "dataset: " << dataset << '\n';
  for (const auto& [k, v] : fields) out << k << ": " << v << '\n';
  for (const std::string& name : file_names) {
    const std::string f = (fs::path(dir) / name).string();
    out << "file: " << name << ' ' << fs::file_size(f) << ' ' << sha256_file(f) << '\n';
  }
  if (!out.flush()) throw std::runtime_error("write error on " + path);
}

json::Value file_record(const std::string& path, const Manifest& manifest) {
  const uint64_t bytes = fs::file_size(path);
  const std::string sha = sha256_file(path);
  const std::string name = fs::path(path).filename().string();
  const auto it = manifest.files.find(name);
  const bool listed = it != manifest.files.end() &&
                      fs::equivalent(fs::absolute(path).parent_path(), manifest.dir);
  if (listed && (it->second.bytes != bytes || it->second.sha256 != sha)) {
    throw std::runtime_error("'" + path + "' does not match its MANIFEST entry (size " +
                             std::to_string(bytes) + ", sha256 " + sha + ")");
  }
  json::Value out = json::Value::object();
  out.set("path", fs::absolute(path).string());
  out.set("bytes", bytes);
  out.set("sha256", sha);
  out.set("in_manifest", listed);
  return out;
}

}  // namespace bench
