#pragma once

namespace fl {
namespace app {
namespace objdet {

#include "flashlight/app/objdet/nn/Transformer.h"
class MLP : public Sequential {

public: 
MLP(const int32_t inputDim,
      const int32_t hiddenDim,
      const int32_t outputDim,
      const int32_t numLayers)
  {
    add(Linear(inputDim, hiddenDim));
    for(int i = 1; i < numLayers - 1; i++) {
      add(ReLU());
      add(Linear(hiddenDim, hiddenDim));
    }
    add(ReLU());
    add(Linear(hiddenDim, outputDim));
  }

private:
   MLP() = default;
   FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

class Detr : public Container {

public:


  Detr(
      //std::shared_ptr<Module> backbone,
      std::shared_ptr<Transformer> transformer,
      const int32_t hiddenDim,
      const int32_t numClasses,
      const int32_t numQueries,
      const bool auxLoss) :
    //backbone_(backbone),
    transformer_(transformer),
    numClasses_(numClasses),
    numQueries_(numQueries),
    auxLoss_(auxLoss),
    classEmbed_(std::make_shared<Linear>(hiddenDim, numClasses + 1)),
    bboxEmbed_(std::make_shared<MLP>(hiddenDim, hiddenDim, 4, 3)),
    queryEmbed_(std::make_shared<Embedding>(hiddenDim, numQueries)),
    inputProj_(std::make_shared<Conv2D>(512, hiddenDim, 1, 1)),
    posEmbed_(std::make_shared<PositionalEmbeddingSine>(hiddenDim / 2,
          10000, true, 6.283185307179586f))
  {
    //add(backbone_);
    add(bboxEmbed_);
    add(classEmbed_);
    add(inputProj_);
    add(posEmbed_);
    add(queryEmbed_);
    add(transformer_);
  }

  std::vector<Variable> forward(const std::vector<Variable>& input) {

    //auto backboneFeatures = backbone_->forward(input);
    auto features = input[0];
    auto mask = input[1];
    auto backboneFeatures = input;
    auto inputProjection = inputProj_->forward(features);
    auto posEmbed = posEmbed_->forward({mask})[0];
    //return { inputProjection, posEmbed };
    auto hs = transformer_->forward(
        inputProjection,
        mask,
        queryEmbed_->param(0),
        posEmbed);

    auto outputClasses = classEmbed_->forward(hs[0]);
    auto outputCoord = sigmoid(bboxEmbed_->forward(hs)[0]);

    return { outputClasses, outputCoord };
  }

  std::string prettyString() const {
    // TODO print params
    return "Detection Transformer!";
  }

private:
  Detr() = default;
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
 FL_SAVE_LOAD_WITH_BASE(fl::Container, backbone_, transformer_, classEmbed_, bboxEmbed_, queryEmbed_, posEmbed_, inputProj_)

};

} // end namespace objdet
} // end namespace app
} // end namespace fl
CEREAL_REGISTER_TYPE(fl::app::objdet::Detr)
CEREAL_REGISTER_TYPE(fl::app::objdet::MLP)
