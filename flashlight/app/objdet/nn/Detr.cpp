#include "flashlight/app/objdet/nn/Detr.h"

namespace {

std::shared_ptr<fl::Linear> makeLinear(int inDim, int outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  auto weights = fl::uniform(outDim, inDim, -std, std);
  auto bias = fl::uniform({outDim}, -std, std, f32, true);
  return std::make_shared<fl::Linear>(weights, bias);
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
  add(Linear(inputDim, hiddenDim));
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
      queryEmbed_(std::make_shared<Embedding>(hiddenDim, numQueries)),
      inputProj_(std::make_shared<Conv2D>(2048, hiddenDim, 1, 1)),
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
