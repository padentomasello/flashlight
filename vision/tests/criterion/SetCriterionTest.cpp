#include "vision/criterion/SetCriterion.h"

#include <gtest/gtest.h>

using namespace fl;
using namespace fl::cv;
/*

TEST(SetCriterion, SetCriterionTest1) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 2;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 1;
  std::vector<float> predBoxesVec = {
    2, 2, 3, 3,
    1, 1, 2, 2
  };

  std::vector<float> targetBoxesVec = {
    1, 1, 2, 2,
    2, 2, 3, 3
  };

  std::vector<float> predLogitsVec = {
    1, 2,
    2, 1
  };

  std::vector<float> targetClassVec = {
    0, 1
  };
  std::vector<float> expRowIds = {
    0, 1
  };
  std::vector<float> colRowIds = {
    1, 0
  };
  auto predBoxes = fl::Variable(af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data()), true);
  auto predLogits = 
    fl::Variable(af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predLogitsVec.data()), true);
  auto targetBoxes = 
    fl::Variable(af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data()), false);
  auto targetClasses = fl::Variable(af::array(1, NUM_TARGETS, NUM_BATCHES, targetClassVec.data()), false);
  auto matcher = HungarianMatcher(1, 1, 1);
  SetCriterion::LossDict losses;
  auto crit = SetCriterion(10, matcher, af::array(), 0.0, losses);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  af_print(loss["loss_giou"].array());
  af_print(loss["loss_ce"].array());
  af_print(loss["loss_bbox"].array());
}

TEST(SetCriterion, SetCriterionBatch) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 2;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 2;
  std::vector<float> predBoxesVec = {
    1, 1, 2, 2,
    2, 2, 3, 3,
    2, 2, 3, 3,
    1, 1, 2, 2
  };

  std::vector<float> targetBoxesVec = { 
    1, 1, 2, 2, 
    2, 2, 3, 3,
    1, 1, 2, 2, 
    2, 2, 3, 3,
  };
  std::vector<float> predLogitsVec = { 2, 1, 2, 1 };
  std::vector<float> targetClassVec = { 0, 0, 0, 0 };
  std::vector<float> expRowIds = { 0, 1, 0, 1 };
  std::vector<float> colRowIds = { 0, 1, 1, 0 };
  auto predBoxes = Variable(af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data()), true);
  auto predLogits = Variable(af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predLogitsVec.data()), true);
  auto targetBoxes = Variable(af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data()), true);
  auto targetClasses = Variable(af::array(1, NUM_TARGETS, NUM_BATCHES, targetClassVec.data()), true);
  auto matcher = HungarianMatcher(1, 1, 1);
  SetCriterion::LossDict losses;
  auto crit = SetCriterion(10, matcher, af::array(), 0.0, losses);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  af_print(loss["loss_giou"].array());
  af_print(loss["loss_ce"].array());
  af_print(loss["loss_bbox"].array());
}

TEST(SetCriterion, SetCriterionBatchFiltered) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 1;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 2;
  std::vector<float> predBoxesVec = {
    1, 1.1, 2, 2,
    2, 2, 3, 3,
    2, 2, 3, 3,
    1, 1, 2, 2
  };

  std::vector<float> targetBoxesVec = { 
    1, 1, 2, 2, 
    2, 2, 3, 3,
  };
  std::vector<float> predLogitsVec = { 2, 1, 2, 1 };
  std::vector<float> targetClassVec = { 0, 0, 0, 0 };
  std::vector<float> expRowIds = { 0, 1, 0, 1 };
  std::vector<float> colRowIds = { 0, 1, 1, 0 };
  auto predBoxes = Variable(af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data()), true);
  auto predLogits = Variable(af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predLogitsVec.data()), true);
  auto targetBoxes = Variable(af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data()), true);
  auto targetClasses = Variable(af::array(1, NUM_TARGETS, NUM_BATCHES, targetClassVec.data()), true);
  auto matcher = HungarianMatcher(1, 1, 1);
  SetCriterion::LossDict losses;
  auto crit = SetCriterion(10, matcher, af::array(), 0.0, losses);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  af_print(loss["loss_giou"].array());
  af_print(loss["loss_ce"].array());
  af_print(loss["loss_bbox"].array());
}

TEST(SetCriterion, ClassPreditions) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 2;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 1;
  std::vector<float> predBoxesVec = {
    1, 1, 2, 2,
    1, 1, 2, 2
  };

  std::vector<float> targetBoxesVec = {
    1, 1, 2, 2,
    1, 1, 2, 2,
  };

  std::vector<float> predLogitsVec = {
    1, 0, // Class 0
    0, 1 // Class 1
  };

  std::vector<float> targetClassVec = {
    1, 0
  };
  std::vector<float> expRowIds = {
    0, 1
  };
  std::vector<float> colRowIds = {
    1, 0
  };
  auto predBoxes = fl::Variable(af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data()), true);
  auto predLogits = 
    fl::Variable(af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predLogitsVec.data()), true);
  auto targetBoxes = 
    fl::Variable(af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data()), false);
  auto targetClasses = fl::Variable(af::array(1, NUM_TARGETS, NUM_BATCHES, targetClassVec.data()), false);
  auto matcher = HungarianMatcher(1, 1, 1);
  SetCriterion::LossDict losses;
  auto crit = SetCriterion(10, matcher, af::array(), 0.0, losses);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  af_print(loss["loss_giou"].array());
  af_print(loss["loss_ce"].array());
  af_print(loss["loss_bbox"].array());
}
*/

TEST(SetCriterion, PytorchRepro) {
  const int NUM_CLASSES = 80;
  const int NUM_TARGETS = 1;
  const int NUM_PREDS = 1;
  const int NUM_BATCHES = 1;
  std::vector<float> predBoxesVec = {
    2, 2, 3, 3
  };

  std::vector<float> targetBoxesVec = {
    2, 2, 3, 3
  };

  std::vector<float> predLogitsVec = {
    1 // Class 0
  };

  std::vector<float> targetClassVec = {
    1
  };
  auto predBoxes = fl::Variable(af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data()), true);
  auto predLogits = 
    fl::Variable(af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predLogitsVec.data()), true);
  std::vector<fl::Variable> targetBoxes = { 
    fl::Variable(af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data()), false) 
  };
  std::vector<fl::Variable>  targetClasses = {
  fl::Variable(af::array(1, NUM_TARGETS, NUM_BATCHES, targetClassVec.data()), false) 
  };
  auto matcher = HungarianMatcher(1, 1, 1);
  SetCriterion::LossDict losses;
  auto crit = SetCriterion(10, matcher, af::array(), 0.0, losses);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  for(auto it : loss) {
    std::cout << it.first;
    af_print(it.second.array());
  }
}
