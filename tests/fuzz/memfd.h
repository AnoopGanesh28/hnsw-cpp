#pragma once

// An anonymous in-memory file holding a fuzz input, for code that only reads
// from paths. The file is reachable as /proc/self/fd/<fd> until destroyed.

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace fuzz {

class MemFile {
 public:
  MemFile(const uint8_t* data, size_t size) : fd_(::memfd_create("fuzz_input", MFD_CLOEXEC)) {
    if (fd_ < 0) die("memfd_create");
    while (size > 0) {
      const ssize_t w = ::write(fd_, data, size);
      if (w < 0 && errno == EINTR) continue;
      if (w <= 0) die("write");
      data += w;
      size -= static_cast<size_t>(w);
    }
    path_ = "/proc/self/fd/" + std::to_string(fd_);
  }
  ~MemFile() { ::close(fd_); }
  MemFile(const MemFile&) = delete;
  MemFile& operator=(const MemFile&) = delete;
  const std::string& path() const { return path_; }

 private:
  // A harness failure, not a finding in the code under test.
  [[noreturn]] static void die(const char* what) {
    std::perror(what);
    std::abort();
  }

  int fd_;
  std::string path_;
};

}  // namespace fuzz
