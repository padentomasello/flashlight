#pragma once 

#include <arrayfire.h>

#include "flashlight/fl/dataset/datasets.h"

namespace fl {
namespace app {
namespace objdet {

using TransformAllFunction = std::function<std::vector<af::array>(const std::vector<af::array>&)>;

class TransformAllDataset : public Dataset {

public:

  TransformAllDataset(
    std::shared_ptr<const Dataset> dataset,
    TransformAllFunction fn) : dataset_(dataset), fn_(fn) {};

  std::vector<af::array> get(const int64_t idx) const override {
    return fn_(dataset_->get(idx));
  }

  int64_t size() const override { return dataset_->size(); };

private:
  std::shared_ptr<const Dataset> dataset_;
  const TransformAllFunction fn_;

};

} // namespace objdet
} // namespace app
} // namespace fl
