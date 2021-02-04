#pragma once

#include <arrayfire.h>

namespace fl {
namespace app {
namespace objdet {

std::vector<af::array> crop(
    const std::vector<af::array>& in,
    int x,
    int y,
    int tw,
    int th);

std::vector<af::array> hflip(
    const std::vector<af::array>& in);


} // namespace objdet
} // namespace app
} // namespace fl
