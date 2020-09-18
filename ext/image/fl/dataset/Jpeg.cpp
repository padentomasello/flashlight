#include "flashlight/ext/image/af/Jpeg.h"
#include "flashlight/ext/image/fl/dataset/Utils.h"

namespace fl {
namespace ext {
namespace image {

std::shared_ptr<Dataset> jpegLoader(std::vector<std::string> fps) {
  return std::make_shared<Loader<std::string>>(fps,
    [](const std::string& fp) {
      std::vector<af::array> result = { loadJpeg(fp) };
      return result;
  });
}

} // namespace image
} // namespace ext
} // namespace fl