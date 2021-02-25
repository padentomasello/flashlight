/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/image/fl/models/ResnetFrozenBatchNorm.h"

namespace fl {
namespace ext {
namespace image {

namespace {

Conv2D conv3x3(int inC, int outC, int stride, int groups) {
  const auto pad = PaddingMode::SAME;
  return Conv2D(
      inC, outC, 3, 3, stride, stride, pad, pad, 1, 1, false, groups);
}

Conv2D conv1x1(int inC, int outC, int stride, int groups) {
  const auto pad = PaddingMode::SAME;
  return Conv2D(
      inC, outC, 1, 1, stride, stride, pad, pad, 1, 1, false, groups);
}

} // namespace

ConvFrozenBnAct::ConvFrozenBnAct() = default;

ConvFrozenBnAct::ConvFrozenBnAct(
    const int inC,
    const int outC,
    const int kw,
    const int kh,
    const int sx,
    const int sy,
    bool bn,
    bool act) {
  const auto pad = PaddingMode::SAME;
  const bool bias = !bn;
  add(std::make_shared<fl::Conv2D>(
      inC, outC, kw, kh, sx, sy, pad, pad, 1, 1, bias));
  if (bn) {
    add(std::make_shared<fl::FrozenBatchNorm>(2, outC));
  }
  if (act) {
    add(std::make_shared<fl::ReLU>());
  }
}

ResNetBlockFrozenBn::ResNetBlockFrozenBn() = default;

ResNetBlockFrozenBn::ResNetBlockFrozenBn(const int inC, const int outC, const int stride) {
  add(std::make_shared<Conv2D>(conv3x3(inC, outC, stride, 1)));
  add(std::make_shared<FrozenBatchNorm>(FrozenBatchNorm(2, outC)));
  add(std::make_shared<ReLU>());
  add(std::make_shared<Conv2D>(conv3x3(outC, outC, 1, 1)));
  add(std::make_shared<FrozenBatchNorm>(FrozenBatchNorm(2, outC)));
  add(std::make_shared<ReLU>());
  if (inC != outC || stride > 1) {
    Sequential downsample;
    downsample.add(conv1x1(inC, outC, stride, 1));
    downsample.add(FrozenBatchNorm(2, outC));
    add(downsample);
  }
}

ResNetBottleneckBlockFrozenBn::ResNetBottleneckBlockFrozenBn() = default;

ResNetBottleneckBlockFrozenBn::ResNetBottleneckBlockFrozenBn(
    const int inC,
    const int planes,
    const int stride) {
  const int expansionFactor = 4;
  add(std::make_shared<Conv2D>(conv1x1(inC, planes, 1, 1)));
  add(std::make_shared<FrozenBatchNorm>(FrozenBatchNorm(2, planes)));
  add(std::make_shared<ReLU>());
  add(std::make_shared<Conv2D>(conv3x3(planes, planes, stride, 1)));
  add(std::make_shared<FrozenBatchNorm>(FrozenBatchNorm(2, planes)));
  add(std::make_shared<ReLU>());
  add(std::make_shared<Conv2D>(
      conv1x1(planes, planes * expansionFactor, 1, 1)));
  add(std::make_shared<FrozenBatchNorm>(
      FrozenBatchNorm(2, planes * expansionFactor)));
  add(std::make_shared<ReLU>());
  if (inC != planes * expansionFactor || stride > 1) {
    Sequential downsample;
    downsample.add(conv1x1(inC, planes * expansionFactor, stride, 1));
    downsample.add(FrozenBatchNorm(2, planes * expansionFactor));
    add(downsample);
  }
}

std::vector<fl::Variable> ResNetBottleneckBlockFrozenBn::forward(
    const std::vector<fl::Variable>& inputs) {
  auto c1 = module(0);
  auto bn1 = module(1);
  auto relu1 = module(2);
  auto c2 = module(3);
  auto bn2 = module(4);
  auto relu2 = module(5);
  auto c3 = module(6);
  auto bn3 = module(7);
  auto relu3 = module(8);

  std::vector<fl::Variable> out;
  out = c1->forward(inputs);
  out = bn1->forward(out);

  out = relu1->forward(out);

  out = c2->forward(out);
  out = bn2->forward(out);
  out = relu2->forward(out);

  out = c3->forward(out);
  out = bn3->forward(out);

  std::vector<fl::Variable> shortcut;
  if (modules().size() > 9) {
    shortcut = module(9)->forward(inputs);
  } else {
    shortcut = inputs;
  }
  return relu3->forward({out[0] + shortcut[0]});
}

std::string ResNetBottleneckBlockFrozenBn::prettyString() const {
  return "ResNetBottleneckBlockFrozenBn";
}

std::vector<fl::Variable> ResNetBlockFrozenBn::forward(
    const std::vector<fl::Variable>& inputs) {
  auto c1 = module(0);
  auto bn1 = module(1);
  auto relu1 = module(2);
  auto c2 = module(3);
  auto bn2 = module(4);
  auto relu2 = module(5);
  std::vector<fl::Variable> out;
  out = c1->forward(inputs);
  out = bn1->forward(out);
  out = relu1->forward(out);
  out = c2->forward(out);
  out = bn2->forward(out);

  std::vector<fl::Variable> shortcut;
  if (modules().size() > 6) {
    shortcut = module(6)->forward(inputs);
  } else {
    shortcut = inputs;
  }
  return relu2->forward({out[0] + shortcut[0]});
}

std::string ResNetBlockFrozenBn::prettyString() const {
  return "2-Layer ResNetBlockFrozenBn Conv3x3";
}

ResNetBottleneckStageFrozenBn::ResNetBottleneckStageFrozenBn(
    const int inC,
    const int outC,
    const int numBlocks,
    const int stride) {
  add(ResNetBottleneckBlockFrozenBn(inC, outC, stride));
  const int expansionFactor = 4;
  const int inPlanes = outC * expansionFactor;
  for (int i = 1; i < numBlocks; i++) {
    add(ResNetBottleneckBlockFrozenBn(inPlanes, outC));
  }
};

ResNetBottleneckStageFrozenBn::ResNetBottleneckStageFrozenBn() = default;

ResNetStageFrozenBn::ResNetStageFrozenBn() = default;

ResNetStageFrozenBn::ResNetStageFrozenBn(
    const int inC,
    const int outC,
    const int numBlocks,
    const int stride) {
  add(ResNetBlockFrozenBn(inC, outC, stride));
  for (int i = 1; i < numBlocks; i++) {
    add(ResNetBlockFrozenBn(outC, outC));
  }
}
std::shared_ptr<Sequential> resnet34() {
  auto model = std::make_shared<Sequential>();
  // conv1 -> 244x244x3 -> 112x112x64
  model->add(ConvFrozenBnAct(3, 64, 7, 7, 2, 2));
  // maxpool -> 112x122x64 -> 56x56x64
  model->add(Pool2D(3, 3, 2, 2, -1, -1, PoolingMode::MAX));
  //// conv2_x -> 56x56x64 -> 56x56x64
  model->add(ResNetStageFrozenBn(64, 64, 3, 1));
  //// conv3_x -> 56x56x64 -> 28x28x128
  model->add(ResNetStageFrozenBn(64, 128, 4, 2));
  //// conv4_x -> 28x28x128 -> 14x14x256
  model->add(ResNetStageFrozenBn(128, 256, 6, 2));
  //// conv5_x -> 14x14x256 -> 7x7x256
  model->add(ResNetStageFrozenBn(256, 512, 3, 2));
  //// pool 7x7x512 -> 1x1x512
  model->add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));

  model->add(View(af::dim4(512, -1, 1, 1)));
  model->add(Linear(512, 1000));
  return model;

  ////model->add(ConvFrozenBnAct(512, 1000, 1, 1, 1, 1, false, false));
  // model->add(View({1000, -1}));
  // model->add(LogSoftmax());
  return model;
};

std::shared_ptr<Sequential> resnet50() {
  auto model = std::make_shared<Sequential>();
  // conv1 -> 244x244x3 -> 112x112x64
  model->add(ConvFrozenBnAct(3, 64, 7, 7, 2, 2));
  // maxpool -> 112x122x64 -> 56x56x64
  model->add(Pool2D(3, 3, 2, 2, -1, -1, PoolingMode::MAX));
  //// conv2_x -> 56x56x64 -> 56x56x64
  model->add(ResNetBottleneckStageFrozenBn(64, 64, 3, 1));
  //// conv3_x -> 56x56x64 -> 28x28x128
  model->add(ResNetBottleneckStageFrozenBn(64 * 4, 128, 4, 2));
  ////// conv4_x -> 28x28x128 -> 14x14x256
  model->add(ResNetBottleneckStageFrozenBn(128 * 4, 256, 6, 2));
  ////// conv5_x -> 14x14x256 -> 7x7x256
  model->add(ResNetBottleneckStageFrozenBn(256 * 4, 512, 3, 2));
  ////// pool 7x7x512 -> 1x1x512
  // model->add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));

  // model->add(View(af::dim4(512 * 4, -1, 1, 1)));
  // model->add(Linear(512 * 4, 1000));
  return model;
}

// std::shared_ptr<Sequential> resnet34() {
// auto model = std::make_shared<Sequential>();
//// conv1 -> 244x244x3 -> 112x112x64
// model->add(ConvFrozenBnAct(3, 64, 7, 7, 2, 2));
//// maxpool -> 112x122x64 -> 56x56x64
// model->add(Pool2D(3, 3, 2, 2, -1, -1, PoolingMode::MAX));
//// conv2_x -> 56x56x64 -> 56x56x64
// model->add(ResNetStageFrozenBn(64, 64, 3, 1));
//// conv3_x -> 56x56x64 -> 28x28x128
// model->add(ResNetStageFrozenBn(64, 128, 4, 2));
//// conv4_x -> 28x28x128 -> 14x14x256
// model->add(ResNetStageFrozenBn(128, 256, 6, 2));
//// conv5_x -> 14x14x256 -> 7x7x256
// model->add(ResNetStageFrozenBn(256, 512, 3, 2));
//// pool 7x7x512 -> 1x1x512
// model->add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));

// model->add(View(af::dim4(512, -1, 1, 1)));
// model->add(Linear(512, 1000));

////model->add(ConvFrozenBnAct(512, 1000, 1, 1, 1, 1, false, false));
// model->add(View({1000, -1}));
////model->add(LogSoftmax());
// return model;
//};

} // namespace image
} // namespace ext
} // namespace fl
