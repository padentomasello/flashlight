#include "flashlight/app/objdet/nn/Detr.h"

namespace {

double calculate_gain(double negativeSlope) {
    return std::sqrt(2.0 / (1 + std::pow(negativeSlope, 2)));
}

std::shared_ptr<fl::Linear> makeLinear(int inDim, int outDim) {
  int fanIn = inDim;
  float gain = calculate_gain(std::sqrt(5.0));
  float std = gain / std::sqrt(fanIn);
  float bound = std::sqrt(3.0) * std;
  auto w = fl::uniform(outDim, inDim, -bound, bound, f32, true);
  bound = 1.0 / std::sqrt(fanIn);
  auto b = fl::uniform(af::dim4(outDim), -bound, bound, af::dtype::f32, true);
  return std::make_shared<fl::Linear>(w, b);
}

std::shared_ptr<fl::Conv2D> makeConv2D(int inDim, int outDim, int wx, int wy) {
  int fanIn = wx * wy * inDim;
  float gain = calculate_gain(std::sqrt(5.0f));
  float std = gain / std::sqrt(fanIn);
  float bound = std::sqrt(3.0f) * std;
  auto w = fl::uniform({ wx, wy, inDim, outDim}, -bound, bound, f32, true);
  bound = 1.0f / std::sqrt(fanIn);
  auto b = fl::uniform(af::dim4(1, 1, outDim, 1), -bound, bound, af::dtype::f32, true);
  return std::make_shared<fl::Conv2D>(w, b, 1, 1);
}

} // namespace

namespace fl {
namespace app {
namespace objdet {

MLP::MLP() = default;

MLP::MLP(
    const int32_t inputDim,
    const int32_t hiddenDim,
    const int32_t outputDim,
    const int32_t numLayers) {
  add(makeLinear(inputDim, hiddenDim));
  for (int i = 1; i < numLayers - 1; i++) {
    add(ReLU());
    add(makeLinear(hiddenDim, hiddenDim));
  }
  add(ReLU());
  add(makeLinear(hiddenDim, outputDim));
}

Detr::Detr() = default;
Detr::Detr(
    std::shared_ptr<Transformer> transformer,
    std::shared_ptr<Module> backbone,
    const int32_t hiddenDim,
    const int32_t numClasses,
    const int32_t numQueries,
    const bool auxLoss)
    : transformer_(transformer),
      backbone_(backbone),
      numClasses_(numClasses),
      numQueries_(numQueries),
      auxLoss_(auxLoss),
      classEmbed_(makeLinear(hiddenDim, numClasses + 1)),
      bboxEmbed_(std::make_shared<MLP>(hiddenDim, hiddenDim, 4, 3)),
      queryEmbed_(std::make_shared<Embedding>(fl::normal({hiddenDim, numQueries}))),
      inputProj_(makeConv2D(2048, hiddenDim, 1, 1)),
      posEmbed_(std::make_shared<PositionalEmbeddingSine>(
          hiddenDim / 2,
          10000,
          true,
          6.283185307179586f)) {
  add(transformer_);
  add(classEmbed_);
  add(bboxEmbed_);
  add(queryEmbed_);
  add(inputProj_);
  add(backbone_);
  add(posEmbed_);
}

std::vector<Variable> Detr::forward(const std::vector<Variable>& input) {
  auto features = backbone_->forward({input[0]})[1];
  fl::Variable mask = fl::Variable(
      af::resize(
          input[1].array(),
          features.dims(0),
          features.dims(1),
          AF_INTERP_NEAREST),
      true);
  auto backboneFeatures = input;
  auto inputProjection = inputProj_->forward(features);
  auto posEmbed = posEmbed_->forward({mask})[0];
  auto hs = transformer_->forward(
      inputProjection, mask, queryEmbed_->param(0), posEmbed);

  auto outputClasses = classEmbed_->forward(hs[0]);
  auto outputCoord = sigmoid(bboxEmbed_->forward(hs)[0]);

  return {outputClasses, outputCoord};
}

std::string Detr::prettyString() const {
  return "Detection Transformer";
}

std::vector<fl::Variable> Detr::paramsWithoutBackbone() {
  std::vector<fl::Variable> results;
  std::vector<std::vector<fl::Variable>> childParams;
  childParams.push_back(transformer_->params());
  childParams.push_back(classEmbed_->params());
  childParams.push_back(bboxEmbed_->params());
  childParams.push_back(queryEmbed_->params());
  childParams.push_back(inputProj_->params());
  for (auto params : childParams) {
    results.insert(results.end(), params.begin(), params.end());
  }
  return results;
}

std::vector<fl::Variable> Detr::backboneParams() {
  return backbone_->params();
}

} // end namespace objdet
} // end namespace app
} // end namespace fl
