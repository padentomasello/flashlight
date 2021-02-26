#pragma once

#include <iostream>
#include "flashlight/ext/image/fl/models/FrozenBatchNorm.h"
#include "flashlight/ext/image/fl/models/ResnetFrozenBatchNorm.h"
#include "flashlight/fl/nn/modules/Container.h"

namespace fl {
namespace app {
namespace objdet {

using namespace fl::ext::image;

class Resnet50Backbone : public Container {
 public:
  Resnet50Backbone()
      : backbone_(std::make_shared<Sequential>()),
        tail_(std::make_shared<Sequential>()) {
    backbone_->add(ConvFrozenBnAct(3, 64, 7, 7, 2, 2));
    // maxpool -> 112x122x64 -> 56x56x64
    backbone_->add(Pool2D(3, 3, 2, 2, -1, -1, PoolingMode::MAX));
    //// conv2_x -> 56x56x64 -> 56x56x64
    backbone_->add(ResNetBottleneckStageFrozenBn(64, 64, 3, 1));
    //// conv3_x -> 56x56x64 -> 28x28x128
    backbone_->add(ResNetBottleneckStageFrozenBn(64 * 4, 128, 4, 2));
    ////// conv4_x -> 28x28x128 -> 14x14x256
    backbone_->add(ResNetBottleneckStageFrozenBn(128 * 4, 256, 6, 2));
    ////// conv5_x -> 14x14x256 -> 7x7x256
    backbone_->add(ResNetBottleneckStageFrozenBn(256 * 4, 512, 3, 2));

     tail_->add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));  
     tail_->add(ConvFrozenBnAct(512 * 4, 1000, 1, 1, 1, 1, false, false));  
     tail_->add(View({1000, -1}));
     tail_->add(LogSoftmax());
     add(backbone_);
     add(tail_);
  }

  std::vector<Variable> forward(const std::vector<Variable>& input) {
    auto features = module(0)->forward(input);
     auto output = module(1)->forward(features);
    return {output[0], features[0]};
  }

  std::string prettyString() const {
    return "Resnet50 Backbone";
  }

 private:
  std::shared_ptr<Sequential> backbone_;
  std::shared_ptr<Sequential> tail_;
  // FL_SAVE_LOAD_WITH_BASE(fl::Container, backbone_, tail_)
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

} // namespace objdet
} // namespace app
} // end namespace fl

CEREAL_REGISTER_TYPE(fl::app::objdet::Resnet50Backbone)
