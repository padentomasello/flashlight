#pragma once

#include <arrayfire.h>
#include <tuple>

#include "flashlight/fl/autograd/Variable.h"

namespace fl {
namespace app {
namespace objdet {

fl::Variable cxcywh2xyxy(const fl::Variable& bboxes);

af::array xyxy2cxcywh(const af::array& bboxes);

typedef Variable (*batchFuncVar_t)(const Variable& lhs, const Variable& rhs);

fl::Variable
cartesian(const fl::Variable& x, const fl::Variable& y, batchFuncVar_t fn);

fl::Variable flatten(const fl::Variable& x, int start, int stop);

Variable generalizedBoxIou(const Variable& bboxes1, const Variable& bboxes2);

std::tuple<fl::Variable, fl::Variable> boxIou(
    const fl::Variable& bboxes1,
    const fl::Variable& bboxes2);

Variable l1Loss(const Variable& input, const Variable& target);

} // namespace objdet
} // namespace app
} // namespace fl
