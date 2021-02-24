#include <gflags/gflags.h>
#include <sstream>

#include "flashlight/ext/common/Runtime.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

namespace fl {
namespace ext {

using fl::lib::format;
using fl::lib::pathsConcat;

//TODO move out of ASR
std::string getRunFile(const std::string& name, int runidx, const std::string& runpath) {
  auto fname = format("%03d_%s", runidx, name.c_str());
  return pathsConcat(runpath, fname);
};

std::string serializeGflags(const std::string& separator) {
  std::stringstream serialized;
  std::vector<gflags::CommandLineFlagInfo> allFlags;
  gflags::GetAllFlags(&allFlags);
  std::string currVal;
  for (auto itr = allFlags.begin(); itr != allFlags.end(); ++itr) {
    gflags::GetCommandLineOption(itr->name.c_str(), &currVal);
    serialized << "--" << itr->name << "=" << currVal << separator;
  }
  return serialized.str();
}

} // end namespace ext
} // end namespace fl
