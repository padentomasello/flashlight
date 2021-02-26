#pragma once

#include "flashlight/app/objdet/nn/PositionalEmbeddingSine.h"
#include "flashlight/app/objdet/nn/Transformer.h"

namespace fl {
namespace app {
namespace objdet {

class MLP : public Sequential {
 public:
  MLP();
  MLP(const int32_t inputDim,
      const int32_t hiddenDim,
      const int32_t outputDim,
      const int32_t numLayers);

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

class Detr : public Container {
 public:
  Detr();
  Detr(
      std::shared_ptr<Transformer> transformer,
      std::shared_ptr<Module> backbone,
      const int32_t hiddenDim,
      const int32_t numClasses,
      const int32_t numQueries,
      const bool auxLoss);

  std::vector<Variable> forward(const std::vector<Variable>& input);

  std::string prettyString() const;

  std::vector<fl::Variable> paramsWithoutBackbone();

  std::vector<fl::Variable> backboneParams();

 private:
  std::shared_ptr<Module> backbone_;
  std::shared_ptr<Transformer> transformer_;
  std::shared_ptr<Linear> classEmbed_;
  std::shared_ptr<MLP> bboxEmbed_;
  std::shared_ptr<Embedding> queryEmbed_;
  std::shared_ptr<PositionalEmbeddingSine> posEmbed_;
  std::shared_ptr<Conv2D> inputProj_;
  int32_t hiddenDim_;
  int32_t numClasses_;
  int32_t numQueries_;
  bool auxLoss_;
  FL_SAVE_LOAD_WITH_BASE(
      fl::Container,
      backbone_,
      transformer_,
      classEmbed_,
      bboxEmbed_,
      queryEmbed_,
      posEmbed_,
      inputProj_)
};

} // end namespace objdet
} // end namespace app
} // end namespace fl
CEREAL_REGISTER_TYPE(fl::app::objdet::Detr)
CEREAL_REGISTER_TYPE(fl::app::objdet::MLP)
