/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/ext/image/fl/models/FrozenBatchNorm.h"
#include "flashlight/fl/nn/nn.h"

namespace fl {
namespace ext {
namespace image {

// Note these are identical to those in Resnet.h. There are a number of ways to
// refactor and consolidate including passing norm factory functions to the
// constructor or templating the class. However, for the sake of keeping
// the default Resnet implementation dead simple, we are recreating a lot
// of functionality here.

class ConvFrozenBnAct : public fl::Sequential {
 public:
  ConvFrozenBnAct();
  explicit ConvFrozenBnAct(
      const int inChannels,
      const int outChannels,
      const int kw,
      const int kh,
      const int sx = 1,
      const int sy = 1,
      bool bn = true,
      bool act = true);

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

class ResNetBlockFrozenBn : public fl::Container {
 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
 public:
  ResNetBlockFrozenBn();
  explicit ResNetBlockFrozenBn(
      const int inChannels,
      const int outChannels,
      const int stride = 1);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  std::string prettyString() const override;
};

class ResNetBottleneckBlockFrozenBn : public fl::Container {
 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
 public:
  ResNetBottleneckBlockFrozenBn();
  explicit ResNetBottleneckBlockFrozenBn(
      const int inChannels,
      const int outChannels,
      const int stride = 1);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  std::string prettyString() const override;
};

class ResNetBottleneckStageFrozenBn : public fl::Sequential {
 public:
  ResNetBottleneckStageFrozenBn();
  explicit ResNetBottleneckStageFrozenBn(
      const int inChannels,
      const int outChannels,
      const int numBlocks,
      const int stride);
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

class ResNetStageFrozenBn : public fl::Sequential {
 public:
  ResNetStageFrozenBn();
  explicit ResNetStageFrozenBn(
      const int inChannels,
      const int outChannels,
      const int numBlocks,
      const int stride);
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

std::shared_ptr<Sequential> resnet34();
std::shared_ptr<Sequential> resnet50();

} // namespace image
} // namespace ext
} // namespace fl
CEREAL_REGISTER_TYPE(fl::ext::image::ConvFrozenBnAct)
CEREAL_REGISTER_TYPE(fl::ext::image::ResNetBlockFrozenBn)
CEREAL_REGISTER_TYPE(fl::ext::image::ResNetStageFrozenBn)
CEREAL_REGISTER_TYPE(fl::ext::image::ResNetBottleneckBlockFrozenBn)
CEREAL_REGISTER_TYPE(fl::ext::image::ResNetBottleneckStageFrozenBn)
