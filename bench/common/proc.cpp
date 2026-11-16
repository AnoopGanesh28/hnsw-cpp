#include "proc.h"

#include <sys/wait.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace bench {

int64_t read_status_kib(const std::string& path, const std::string& key) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read '" + path + "'");
  const std::string prefix = key + ":";
  std::string line;
  while (std::getline(in, line)) {
    if (line.compare(0, prefix.size(), prefix) != 0) continue;
    const char* p = line.c_str() + prefix.size();
    char* end = nullptr;
    errno = 0;
    const long long v = std::strtoll(p, &end, 10);
    const std::string rest = end ? std::string(end) : std::string();
    if (end == p || errno != 0 || v < 0 || rest != " kB") {
      throw std::runtime_error("'" + path + "': malformed " + key + " line '" + line + "'");
    }
    return v;
  }
  throw std::runtime_error("'" + path + "': no " + key + " line");
}

}  // namespace bench

namespace bench {

std::string run_command(const std::string& command) {
  FILE* pipe = ::popen(command.c_str(), "r");
  if (!pipe) throw std::runtime_error("cannot run '" + command + "'");
  std::string out;
  char buf[4096];
  size_t got = 0;
  while ((got = std::fread(buf, 1, sizeof(buf), pipe)) > 0) out.append(buf, got);
  const int status = ::pclose(pipe);
  if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error("command failed: '" + command + "'");
  }
  return out;
}

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (const char c : s) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  return out + "'";
}

}  // namespace bench
