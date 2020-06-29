/**
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <exception>
#include <iomanip>

#include <gflags/gflags.h>

#include "flashlight/dataset/datasets.h"
#include "flashlight/meter/meters.h"
#include "flashlight/optim/optim.h"
#include "vision/dataset/ImagenetUtils.h"
#include "vision/dataset/Coco.h"
#include "vision/dataset/BoxUtils.h"
#include "vision/dataset/Transforms.h"
#include "vision/dataset/Utils.h"
#include "vision/models/Resnet.h"
#include "vision/nn/PositionalEmbeddingSine.h"
#include "vision/nn/Transformer.h"
#include "vision/criterion/SetCriterion.h"

DEFINE_string(data_dir, "/datasets01_101/imagenet_full_size/061417/", "Directory of imagenet data");
DEFINE_double(lr, 0.1f, "Learning rate");
DEFINE_double(momentum, 0.9f, "Momentum");

DEFINE_double(wd, 1e-4f, "Weight decay");
DEFINE_uint64(epochs, 50, "Epochs");
DEFINE_int64(
    world_rank,
    0,
    "rank of the process (Used if rndv_filepath is not empty)");
DEFINE_int64(
    world_size,
    1,
    "total number of the process (Used if rndv_filepath is not empty)");
DEFINE_string(
    rndv_filepath,
    "/tmp/",
    "Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");
DEFINE_uint64(batch_size, 256, "Total batch size across all gpus");
DEFINE_string(checkpointpath, "/tmp/model", "Checkpointing prefix path");
DEFINE_int64(checkpoint, -1, "Load from checkpoint");


using namespace fl;
using namespace fl::cv;
using namespace cv::dataset;

// TODO Refactor
const int32_t backboneChannels = 512;

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
};

class Detr : public Container {

public:

  Detr(
      std::shared_ptr<Module> backbone,
      std::shared_ptr<Transformer> transformer,
      const int32_t hiddenDim,
      const int32_t numClasses,
      const int32_t numQueries,
      const bool auxLoss) :
    backbone_(backbone),
    transformer_(transformer),
    numClasses_(numClasses),
    numQueries_(numQueries),
    auxLoss_(auxLoss),
    classEmbed_(std::make_shared<Linear>(hiddenDim, numClasses + 1)),
    bboxEmbed_(std::make_shared<MLP>(hiddenDim, hiddenDim, 4, 3)),
    queryEmbed_(std::make_shared<Embedding>(hiddenDim, numQueries)),
    inputProj_(std::make_shared<Conv2D>(512, hiddenDim, 1, 1)),
    posEmbed_(std::make_shared<PositionalEmbeddingSine>(hiddenDim / 2,
          10000, false, 0.0f))
  {
    add(backbone_);
    add(transformer_);
    add(classEmbed_);
    add(queryEmbed_);
    add(inputProj_);
  }

  std::vector<Variable> forward(const std::vector<Variable>& input) {

    auto backboneFeatures = backbone_->forward(input);
    auto inputProjection = inputProj_->forward(backboneFeatures[0]);
    auto posEmbed = posEmbed_->forward(backboneFeatures[0]);
    auto hs = transformer_->forward(
        inputProjection,
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

};

int main(int argc, char** argv) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const std::string label_path = FLAGS_data_dir + "labels.txt";
  const std::string train_list = FLAGS_data_dir + "train";
  const std::string val_list = FLAGS_data_dir + "val";

  /////////////////////////
  // Setup distributed training
  ////////////////////////

  //////////////////////////
  //  Create datasets
  /////////////////////////
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  std::vector<ImageTransform> train_transforms = {
      // randomly resize shortest side of image between 256 to 480 for scale 
      // invariance
      randomResizeTransform(256, 480),
      randomCropTransform(224, 224),
      normalizeImage(mean, std),
      // Randomly flip image with probability of 0.5
      horizontalFlipTransform(0.5)
  };
  std::vector<ImageTransform> val_transforms = {
      // Resize shortest side to 256, then take a center crop
      resizeTransform(256),
      centerCropTransform(224),
      normalizeImage(mean, std)
  };

  const int32_t modelDim = 64;
  const int32_t numHeads = 6;
  const int32_t numEncoderLayers = 6;
  const int32_t numDecoderLayers = 6;
  const int32_t mlpDim = modelDim;
  // TODO check this is correct
  const int32_t hiddenDim = modelDim;
  const int32_t numClasses = 80;
  const int32_t numQueries = 100;
  const float pDropout = 0.0;
  const bool auxLoss = false;

  std::shared_ptr<Module> backbone; 
  backbone = std::make_shared<Sequential>(resnet34());
  auto transformer = std::make_shared<Transformer>(
      modelDim,
      numHeads,
      numEncoderLayers,
      numDecoderLayers,
      mlpDim,
      pDropout);

  auto detr = Detr(
      backbone,
      transformer,
      hiddenDim,
      numClasses,
      numQueries,
      auxLoss);


  auto matcher = HungarianMatcher(1, 1, 1);
  SetCriterion::LossDict losses;
  auto criterion = SetCriterion(
      numClasses,
      matcher,
      af::array(),
      0.0,
      losses);



  const int64_t batch_size_per_gpu = FLAGS_batch_size / FLAGS_world_size;
  const int64_t prefetch_threads = 10;
  const int64_t prefetch_size = FLAGS_batch_size;
  std::string coco_list = "/private/home/padentomasello/data/coco-mini/train.lst";
  auto coco = cv::dataset::coco(coco_list, val_transforms);
  //SGDOptimizer opt(detr.params(), FLAGS_lr, FLAGS_momentum, FLAGS_wd);
  AdamOptimizer opt(detr.params(), FLAGS_lr);

  for(int i = 0; i < FLAGS_epochs; i++) {

  AverageValueMeter accum_loss_meter;
  AverageValueMeter bbox_loss_meter;
  AverageValueMeter giou_loss_meter;
  AverageValueMeter ce_loss_meter;
  std::unordered_map<std::string, AverageValueMeter> meters;
  for(auto& sample : *coco) {
    std::cout << " Sample done" << std::endl;
    auto images =  { fl::Variable(sample.images, false) };
    auto output = detr.forward(images);

    std::vector<Variable> targetBoxes(sample.target_boxes.size());
    std::vector<Variable> targetClasses(sample.target_labels.size());

    std::transform(
        sample.target_boxes.begin(), sample.target_boxes.end(),
        targetBoxes.begin(),
        [](const af::array& in) { return fl::Variable(in, false); });

    std::transform(
        sample.target_labels.begin(), sample.target_labels.end(),
        targetClasses.begin(),
        [](const af::array& in) { return fl::Variable(in, false); });

    auto loss = criterion.forward(
        output[1],
        output[0],
        targetBoxes,
        targetClasses);
    auto accumLoss = fl::Variable(af::constant(0, 1), true);
    for(auto losses : loss) {
      //const char* name = losses.first.c_str();
      //const af::array arr = losses.second.array();
      //af::print(name, arr);
      meters[losses.first].add(losses.second.array());
      accumLoss = losses.second + accumLoss;
    }
    meters["sum"].add(accumLoss.array());
    accumLoss.backward();
    //af_print(accumLoss.array());
    //af_print(loss["loss_giou"].array());
    opt.step();
    opt.zeroGrad();
  }
  std::cout << "Batch: " << i;
  for(auto meter : meters) {
    std::cout << " " << meter.first << ": " << meter.second.value()[0];
    meter.second.reset();
  }
  std::cout << std::endl;
}
}
