// Runs saved fuzz inputs through a target's LLVMFuzzerTestOneInput without
// libFuzzer, so the seeds, the corpus and past crashes are regression tests in
// every build (including the sanitizer builds).
//
// Usage: replay_<target> <file or directory>...
// Directories contribute their regular files (not recursive; names starting
// with '.' are skipped). Fails if a path does not exist or nothing was
// replayed; a finding aborts the process, as under libFuzzer.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  std::vector<fs::path> inputs;
  for (int i = 1; i < argc; ++i) {
    const fs::path arg = argv[i];
    if (fs::is_regular_file(arg)) {
      inputs.push_back(arg);
    } else if (fs::is_directory(arg)) {
      for (const auto& e : fs::directory_iterator(arg)) {
        if (e.is_regular_file() && e.path().filename().string()[0] != '.') inputs.push_back(e.path());
      }
    } else {
      std::fprintf(stderr, "replay: no such file or directory: %s\n", argv[i]);
      return 1;
    }
  }
  std::sort(inputs.begin(), inputs.end());
  for (const fs::path& p : inputs) {
    std::ifstream in(p, std::ios::binary);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(in),
                                  std::istreambuf_iterator<char>()};
    if (!in.eof() && !in) {
      std::fprintf(stderr, "replay: cannot read %s\n", p.c_str());
      return 1;
    }
    // libFuzzer never passes a null pointer, even for an empty input.
    const uint8_t empty = 0;
    const uint8_t* data = bytes.empty() ? &empty : reinterpret_cast<const uint8_t*>(bytes.data());
    LLVMFuzzerTestOneInput(data, bytes.size());
  }
  std::printf("replayed %zu inputs\n", inputs.size());
  return inputs.empty() ? 1 : 0;
}
