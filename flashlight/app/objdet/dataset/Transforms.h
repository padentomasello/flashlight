#pragma once

#include <arrayfire.h>

#include "flashlight/app/objdet/dataset/TransformAllDataset.h"

namespace fl {
namespace app {
namespace objdet {

enum DatasetIndices {
  ImageIdx = 0,
  TargetSizeIdx = 1,
  ImageIdIdx = 2,
  OriginalSizeIdx = 3,
  BboxesIdx = 4,
  ClassesIdx = 5
};


std::vector<af::array> crop(
    const std::vector<af::array>& in,
    int x,
    int y,
    int tw,
    int th);

std::vector<af::array> hflip(const std::vector<af::array>& in);

std::vector<af::array> Normalize(const std::vector<af::array>& in);

std::vector<af::array> randomResize(std::vector<af::array> inputs, int size, int maxsize);

TransformAllFunction randomSelect(std::vector<TransformAllFunction> fns);

TransformAllFunction randomSizeCrop(int minSize, int maxSize);

TransformAllFunction randomResize(std::vector<int> sizes, int maxsize);

TransformAllFunction randomHorizontalFlip(float p);

TransformAllFunction compose(std::vector<TransformAllFunction> fns);

} // namespace objdet
} // namespace app
} // namespace fl
