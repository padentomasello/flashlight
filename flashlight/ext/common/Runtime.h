#include <iostream>

namespace fl {
namespace ext {
/**
 * Get a certain checkpoint by `runidx`.
 */
std::string
getRunFile(const std::string& name, int runidx, const std::string& runpath);

/**
 * Serialize gflags into a buffer.
 */
std::string serializeGflags(const std::string& separator = "\n");

} // namespace ext
} // namespace fl
