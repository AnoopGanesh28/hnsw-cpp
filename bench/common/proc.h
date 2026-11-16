#pragma once

#include <cstdint>
#include <string>

namespace bench {

// Value of a "Key:   <number> kB" line in a file in the format of
// /proc/<pid>/status, in KiB. Throws std::runtime_error if the file cannot be
// read, the key is absent, or its line is not "<number> kB".
int64_t read_status_kib(const std::string& path, const std::string& key);

// Peak resident set size of this process so far (VmHWM in /proc/self/status).
inline int64_t vm_hwm_kib() { return read_status_kib("/proc/self/status", "VmHWM"); }

}  // namespace bench

namespace bench {

// Runs `command` with /bin/sh and returns its standard output. Throws
// std::runtime_error if it cannot be started or exits nonzero.
std::string run_command(const std::string& command);

// s quoted for /bin/sh as one word.
std::string shell_quote(const std::string& s);

}  // namespace bench
