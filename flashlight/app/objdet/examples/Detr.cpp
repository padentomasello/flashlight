/**
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <exception>
#include <iomanip>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/app/objdet/common/Defines.h"
#include "flashlight/app/objdet/criterion/SetCriterion.h"
#include "flashlight/app/objdet/dataset/BoxUtils.h"
#include "flashlight/app/objdet/dataset/Coco.h"
#include "flashlight/app/objdet/models/Resnet50Backbone.h"
#include "flashlight/app/objdet/nn/Detr.h"
#include "flashlight/app/objdet/nn/Transformer.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/common/Runtime.h"
#include "flashlight/ext/common/Serializer.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/lib/common/String.h"

using namespace fl;
using namespace fl::ext::image;
using namespace fl::app::objdet;

using fl::ext::Serializer;
using fl::ext::getRunFile;
using fl::ext::serializeGflags;
using fl::lib::fileExists;
using fl::lib::format;
using fl::lib::getCurrentDate;

#define FL_LOG_MASTER(lvl) LOG_IF(lvl, (fl::getWorldRank() == 0))

DEFINE_string(
    data_dir,
    "/private/home/padentomasello/data/coco_new/",
    "Directory of imagenet data");
DEFINE_double(train_lr, 0.0001f, "Learning rate");
DEFINE_uint64(metric_iters, 5, "Print metric every");

DEFINE_double(train_wd, 1e-4f, "Weight decay");
DEFINE_uint64(train_epochs, 300, "train_epochs");
DEFINE_uint64(eval_iters, 1, "Run evaluation every n epochs");
DEFINE_int64(
    distributed_world_rank,
    0,
    "rank of the process (Used if distributed_rndv_filepath is not empty)");
DEFINE_int64(
    distributed_world_size,
    1,
    "total number of the process (Used if distributed_rndv_filepath is not empty)");
DEFINE_string(
    distributed_rndv_filepath,
    "",
    "Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");
DEFINE_bool(distributed_enable, true, "Enable distributed training");
DEFINE_uint64(data_batch_size, 2, "Total batch size across all gpus");

DEFINE_string(
    eval_dir,
    "/private/home/padentomasello/data/coco/output/",
    "Directory to dump images to run evaluation script on");
DEFINE_bool(
    model_pretrained,
    true,
    "Whether to load model_pretrained backbone");
DEFINE_string(
    model_pytorch_init,
    "",
    "Directory to dump images to run evaluation script on");
DEFINE_string(
    flagsfile,
    "",
    "Directory to dump images to run evaluation script on");
DEFINE_string(
    exp_rundir,
    "",
    "Directory to dump images to run evaluation script on");
DEFINE_string(
    eval_script,
    "/private/home/padentomasello/code/flashlight/flashlight/app/objdet/scripts/eval_coco.py",
    "Script to run evaluation on dumped tensors");
DEFINE_string(
    eval_set_env,
    "LD_LIBRARY_PATH=/private/home/padentomasello/usr/lib/:$LD_LIBRARY_PATH ",
    "Set environment");
DEFINE_int64(eval_break, -1, "Break eval after this many iters");
DEFINE_bool(eval_only, false, "Weather to just run eval");
// MIXED PRECISION OPTIONS
DEFINE_bool(
    fl_amp_use_mixed_precision,
    false,
    "[train] Use mixed precision for training - scale loss and gradients up and down "
    "by a scale factor that changes over time. If no fl optim mode is "
    "specified with --fl_optim_mode when passing this flag, automatically "
    "sets the optim mode to O1.");
DEFINE_double(
    fl_amp_scale_factor,
    4096.,
    "[train] Starting scale factor to use for loss scaling "
    " with mixed precision training");
DEFINE_uint64(
    fl_amp_scale_factor_update_interval,
    2000,
    "[train] Update interval for adjusting loss scaling in mixed precision training");
DEFINE_uint64(
    fl_amp_max_scale_factor,
    32000,
    "[train] Maximum value for the loss scale factor in mixed precision training");
DEFINE_string(
    fl_optim_mode,
    "",
    "[train] Sets the flashlight optimization mode. "
    "Optim modes can be O1, O2, or O3.");

// Utility function that overrides flags file with command line arguments
void parseCmdLineFlagsWrapper(int argc, char** argv) {
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, false);
}


bool isBadArray(const af::array& arr) {
  return af::anyTrue<bool>(af::isNaN(arr)) || af::anyTrue<bool>(af::isInf(arr));
}

void getBns(
    std::shared_ptr<fl::Module> module,
    std::vector<std::shared_ptr<fl::Module>>& bns) {
  if (dynamic_cast<fl::FrozenBatchNorm*>(module.get())) {
    bns.push_back(module);
  } else if (dynamic_cast<fl::Container*>(module.get())) {
    for (auto mod : dynamic_cast<fl::Container*>(module.get())->modules()) {
      getBns(mod, bns);
    }
  }
};

void evalLoop(
    std::shared_ptr<Detr> model,
    std::shared_ptr<CocoDataset> dataset) {
  model->eval();
  int idx = 0;
  std::stringstream mkdir_command;
  mkdir_command << "mkdir -p " << FLAGS_eval_dir << fl::getWorldRank();
  system(mkdir_command.str().c_str());
  for (auto& sample : *dataset) {
    std::vector<Variable> input = {fl::Variable(sample.images, false),
                                   fl::Variable(sample.masks, false)};
    auto output = model->forward(input);
    std::stringstream ss;
    ss << FLAGS_eval_dir << fl::getWorldRank() << "/detection" << idx
       << ".array";
    auto outputFile = ss.str();
    int lastLayerIdx = output[0].dims(3) - 1;
    auto scores = output[0].array()(
        af::span, af::span, af::span, af::seq(lastLayerIdx, lastLayerIdx));
    auto bboxes = output[1].array()(
        af::span, af::span, af::span, af::seq(lastLayerIdx, lastLayerIdx));
    af::saveArray(
        "imageSizes", sample.originalImageSizes, outputFile.c_str(), false);
    af::saveArray("imageIds", sample.imageIds, outputFile.c_str(), true);
    af::saveArray("scores", scores, outputFile.c_str(), true);
    af::saveArray("bboxes", bboxes, outputFile.c_str(), true);
    idx++;
  }
  if (FLAGS_distributed_enable) {
    barrier();
  }
  if (fl::getWorldRank() == 0) {
    std::stringstream ss;
    ss << "PYTHONPATH=/private/home/padentomasello/code/detection-transformer/ "
       << FLAGS_eval_set_env << " "
       << "/private/home/padentomasello/.conda/envs/coco/bin/python3.8 "
       << FLAGS_eval_script << " --dir " << FLAGS_eval_dir;
    int numAttempts = 10;
    for (int i = 0; i < numAttempts; i++) {
      int rv = system(ss.str().c_str());
      if (rv == 0) {
        break;
      }
      std::cout << "Eval failed, retrying in 5 seconds" << std::endl;
      sleep(5);
    }
  }
  if (FLAGS_distributed_enable) {
    barrier();
  }
  std::stringstream ss2;
  ss2 << "rm -rf " << FLAGS_eval_dir << fl::getWorldRank() << "/detection*";
  std::cout << "Removing tmp eval files Command: " << ss2.str() << std::endl;
  //system(ss2.str().c_str());
  model->train();
};

int main(int argc, char** argv) {
  fl::init();
  af::info();

  ///////////////////////////
  // Setup train / continue modes
  ///////////////////////////
  int runIdx = 1; // current #runs in this path
  std::string runPath; // current experiment path
  std::string reloadPath; // path to model to reload
  std::string runStatus = argv[1];
  int64_t startEpoch = 0;
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage(
      "Usage: \n " + exec + " train [flags]\n or " + exec +
      " continue [directory] [flags]\n or " + exec +
      " fork [directory/model] [flags]");
  // Saving checkpointing
  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }
  if (runStatus == kTrainMode) {
    parseCmdLineFlagsWrapper(argc, argv);
    runPath = FLAGS_exp_rundir;
  } else if (runStatus == kContinueMode) {
    runPath = argv[2];
    while (fileExists(getRunFile("model_last.bin", runIdx, runPath))) {
      ++runIdx;
    }
    reloadPath = getRunFile("model_last.bin", runIdx - 1, runPath);
    LOG(INFO) << "reload path is " << reloadPath;
    std::unordered_map<std::string, std::string> cfg;
    std::string version;
    Serializer::load(reloadPath, version, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "Invalid config loaded from " << reloadPath;
    }
    LOG(INFO) << "Reading flags from config file " << reloadPath;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    parseCmdLineFlagsWrapper(argc, argv);
    auto epoch = cfg.find(kEpoch);
    if (epoch == cfg.end()) {
      LOG(WARNING) << "Did not find epoch to start from, starting from 0.";
    } else {
      startEpoch = std::stoi(epoch->second);
    }
  } else {
    LOG(FATAL) << gflags::ProgramUsage();
  }

    // flashlight optim mode
  auto flOptimLevel = FLAGS_fl_optim_mode.empty()
      ? fl::OptimLevel::DEFAULT
      : fl::OptimMode::toOptimLevel(FLAGS_fl_optim_mode);
  fl::OptimMode::get().setOptimLevel(flOptimLevel);
  if (FLAGS_fl_amp_use_mixed_precision) {
    // Only set the optim mode to O1 if it was left empty
    LOG(INFO) << "Mixed precision training enabled. Will perform loss scaling.";
    if (FLAGS_fl_optim_mode.empty()) {
      LOG(INFO) << "Mixed precision training enabled with no "
                   "optim mode specified - setting optim mode to O1.";
      fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O1);
    }
  }

  if (runPath.empty()) {
    LOG(FATAL)
        << "'runpath' specified by --exp_rundir, --runname cannot be empty";
  }
  const std::string cmdLine = fl::lib::join(" ", argvs);
  std::unordered_map<std::string, std::string> config = {
      {kProgramName, exec},
      {kCommandLine, cmdLine},
      {kGflags, serializeGflags()},
      // extra goodies
      {kUserName, fl::lib::getEnvVar("USER")},
      {kHostName, fl::lib::getEnvVar("HOSTNAME")},
      {kTimestamp, getCurrentDate() + ", " + getCurrentDate()},
      {kRunIdx, std::to_string(runIdx)},
      {kRunPath, runPath}};

  std::stringstream ss;
    ss << "PYTHONPATH=/private/home/padentomasello/code/detection-transformer/ "
       << FLAGS_eval_set_env << " "
       << "/private/home/padentomasello/.conda/envs/coco/bin/python3.8 "
       << "-c 'import arrayfire as af'";
    std::cout << ss.str() << std::endl;
    system(ss.str().c_str());

  /////////////////////////
  // Setup distributed training
  ////////////////////////
  std::shared_ptr<fl::Reducer> reducer = nullptr;
  if (FLAGS_distributed_enable) {
    fl::ext::initDistributed(
        FLAGS_distributed_world_rank,
        FLAGS_distributed_world_size,
        8,
        FLAGS_distributed_rndv_filepath);

    reducer = std::make_shared<fl::CoalescingReducer>(1.0, true, true);
  }
  const int worldRank = fl::getWorldRank();
  const int worldSize = fl::getWorldSize();

  ////////////////////////////
  // Create models
  ////////////////////////////
  const int32_t modelDim = 256;
  const int32_t numHeads = 8;
  const int32_t numEncoderLayers = 6;
  const int32_t numDecoderLayers = 6;
  const int32_t mlpDim = 2048;
  const int32_t hiddenDim = modelDim;
  const int32_t numClasses = 91;
  const int32_t numQueries = 100;
  const float pDropout = 0.1;
  const bool auxLoss = false;
  std::shared_ptr<Resnet50Backbone> backbone;
  if (FLAGS_model_pretrained) {
    std::string modelPath =
        "/checkpoint/padentomasello/models/resnet50/pretrained2";
    fl::load(modelPath, backbone);
  } else {
    backbone = std::make_shared<Resnet50Backbone>();
  }
  auto transformer = std::make_shared<Transformer>(
      modelDim, numHeads, numEncoderLayers, numDecoderLayers, mlpDim, pDropout);

  auto detr = std::make_shared<Detr>(
      transformer, backbone, hiddenDim, numClasses, numQueries, auxLoss);

  // Trained
  // untrained but initializaed
  if (!FLAGS_model_pytorch_init.empty()) {
    //std::cout << "Loading from pytorch intiialization path"
              //<< FLAGS_model_pytorch_init << std::endl;
  std::string filename =
      "/private/home/padentomasello/scratch/pytorch_testing/detr_initialization.array";

    int paramSize = detr->params().size();
    for (int i = 0; i < paramSize; i++) {
      auto array = af::readArray(filename.c_str(), i + 4);
      if (i == 264) {
        array = af::moddims(array, {1, 1, 256, 1});
      }
      assert(detr->param(i).dims() == array.dims());
      auto fl_mean = af::mean<float>(detr->param(i).array());
      auto pt_mean = af::mean<float>(array);
      auto fl_std = af::stdev<float>(detr->param(i).array());
      auto pt_std = af::stdev<float>(array);
      if(std::abs(fl_mean - pt_mean) > 1e-2 || std::abs(fl_std - pt_std) > 1e-2) {
      std::cout << "i: " << i << " FL mean: " << fl_mean << " PT mean " << pt_mean  << std::endl;
      std::cout << "i: " << i << " FL std: " << fl_std << " PT std " << pt_std << std::endl;
      }
      //assert(std::abs(fl_mean - pt_mean) < 1e-2);
      //assert(std::abs(fl_std - pt_std) < 1e-2);
      detr->setParams(param(array), i);
    }

    std::vector<std::shared_ptr<fl::Module>> bns;
    getBns(detr, bns);

    int i = 0;
    for (auto bn : bns) {
      auto bn_ptr = dynamic_cast<fl::FrozenBatchNorm*>(bn.get());
      bn_ptr->setRunningMean(af::readArray((filename + "running").c_str(), i));
      i++;
      bn_ptr->setRunningVar(af::readArray((filename + "running").c_str(), i));
      i++;
    }
    // std::string modelPath =
    // "/checkpoint/padentomasello/models/detr/from_pytorch";  std::string
    // modelPath =
    // "/checkpoint/padentomasello/models/detr/model_pytorch_initializaition";
    //fl::load(FLAGS_model_pytorch_init, detr);
  }
  detr->train();

  /////////////////////////
  // Build criterion
  /////////////////////////
  const float setCostClass = 1.f;
  const float setCostBBox = 5.f;
  const float setCostGiou = 2.f;
  const float bboxLossCoef = 5.f;
  const float giouLossCoef = 2.f;

  auto matcher = HungarianMatcher(setCostClass, setCostBBox, setCostGiou);

  const std::unordered_map<std::string, float> lossWeightsBase = {
      {"lossCe", 1.f}, {"lossGiou", giouLossCoef}, {"lossBbox", bboxLossCoef}};

  std::unordered_map<std::string, float> lossWeights;
  for (int i = 0; i < numDecoderLayers; i++) {
    for (auto l : lossWeightsBase) {
      std::string key = l.first + "_" + std::to_string(i);
      lossWeights[key] = l.second;
    }
  }
  auto criterion = SetCriterion(numClasses, matcher, lossWeights, 0.1);
  auto weightDict = criterion.getWeightDict();

  ////////////////////
  // Optimizers
  ////////////////////
  const float beta1 = 0.9;
  const float beta2 = 0.999;
  const float epsilon = 1e-8;
  auto opt = std::make_shared<AdamOptimizer>(
      detr->paramsWithoutBackbone(),
      FLAGS_train_lr,
      beta1,
      beta2,
      epsilon,
      FLAGS_train_wd);
  auto opt2 = std::make_shared<AdamOptimizer>(
      detr->backboneParams(),
      FLAGS_train_lr * 0.1,
      beta1,
      beta2,
      epsilon,
      FLAGS_train_wd);
  auto lrScheduler = [&opt, &opt2](int epoch) {
    // Adjust learning rate every 30 epoch after 30
    const float newLr = FLAGS_train_lr * pow(0.1, epoch / 100);
    LOG(INFO) << "Setting learning rate to: " << newLr;
    opt->setLr(newLr);
    opt2->setLr(newLr * 0.1);
  };

  /////////////////////////
  // Create Datasets
  /////////////////////////
  const int64_t data_batch_size_per_gpu = FLAGS_data_batch_size;
  const int64_t prefetch_threads = 10;
  const int64_t prefetch_size = FLAGS_data_batch_size;
  std::string coco_dir = FLAGS_data_dir;
  auto train_ds = std::make_shared<CocoDataset>(
      coco_dir + "train.lst",
      worldRank,
      worldSize,
      data_batch_size_per_gpu,
      prefetch_threads,
      data_batch_size_per_gpu,
      false);

  auto val_ds = std::make_shared<CocoDataset>(
      coco_dir + "val.lst",
      worldRank,
      worldSize,
      data_batch_size_per_gpu,
      prefetch_threads,
      data_batch_size_per_gpu,
      true);

  // Override any initialization if continuing
  if (runStatus == "continue") {
    std::unordered_map<std::string, std::string> cfg; // unused
    std::string version;
    Serializer::load(reloadPath, version, cfg, detr, opt, opt2);
  }

  // Run eval if continueing
  if (startEpoch > 0 || FLAGS_eval_only) {
    detr->eval();
    evalLoop(detr, val_ds);
    detr->train();
    if (FLAGS_eval_only) {
      return 0;
    }
  }
  if (FLAGS_distributed_enable) {
    // synchronize parameters of the model so that the parameters in each
    // process is the same
    fl::allReduceParameters(detr);
    // fl::allReduceParameters(backbone);

    // Add a hook to synchronize gradients of model parameters as they are
    // computed
    fl::distributeModuleGrads(detr, reducer);
  }

  if (FLAGS_fl_amp_use_mixed_precision) {
    // Only set the optim mode to O1 if it was left empty
    FL_LOG_MASTER(INFO)
        << "Mixed precision training enabled. Will perform loss scaling.";
    if (FLAGS_fl_optim_mode.empty()) {
      // fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O1);
      fl::OptimMode::get().setOptimLevel(fl::OptimLevel::DEFAULT);
    }
  }
  unsigned short scaleCounter = 1;
  double scaleFactor =
      FLAGS_fl_amp_use_mixed_precision ? FLAGS_fl_amp_scale_factor : 1.;
  unsigned int kScaleFactorUpdateInterval =
      FLAGS_fl_amp_scale_factor_update_interval;
  unsigned int kMaxScaleFactor = FLAGS_fl_amp_max_scale_factor;

  ////////////////
  // Training loop
  //////////////
  for (int epoch = startEpoch; epoch < FLAGS_train_epochs; epoch++) {
    int idx = 0;
    std::map<std::string, AverageValueMeter> meters;
    std::map<std::string, TimeMeter> timers;

    timers["total"].resume();
    lrScheduler(epoch);
    train_ds->resample();
    for (auto& sample : *train_ds) {
      bool retrySample = false;
      do {
        retrySample = false;
        timers["forward"].resume();
        std::vector<Variable> input = {fl::Variable(sample.images, false),
                                     fl::Variable(sample.masks, false)};
      auto output = detr->forward(input);

      timers["forward"].stop();

      /////////////////////////
      // Criterion
      /////////////////////////
      std::vector<Variable> targetBoxes(sample.target_boxes.size());
      std::vector<Variable> targetClasses(sample.target_labels.size());

      std::transform(
          sample.target_boxes.begin(),
          sample.target_boxes.end(),
          targetBoxes.begin(),
          [](const af::array& in) { return fl::Variable(in, false); });

      std::transform(
          sample.target_labels.begin(),
          sample.target_labels.end(),
          targetClasses.begin(),
          [](const af::array& in) { return fl::Variable(in, false); });

      timers["criterion"].resume();

      auto losses =
          criterion.forward(output[1], output[0], targetBoxes, targetClasses);

      auto loss = fl::Variable(af::constant(0, 1), true);
      for (auto l : losses) {
        fl::Variable scaled_loss = weightDict[l.first] * l.second;
        meters[l.first].add(l.second.array());
        meters[l.first + "_weighted"].add(scaled_loss.array());
        loss = scaled_loss + loss;
      }
      meters["sum"].add(loss.array());
      timers["criterion"].stop();

      /////////////////////////
      // Backward and update gradients
      //////////////////////////
      timers["backward"].resume();
      loss.backward();
      timers["backward"].stop();

      if (FLAGS_fl_amp_use_mixed_precision) {
          ++scaleCounter;
          loss = loss * scaleFactor;
      }

      if (FLAGS_distributed_enable) {
        reducer->finalize();
      }

      fl::clipGradNorm(detr->params(), 0.1);
      if (FLAGS_fl_amp_use_mixed_precision) {
        for (auto& p : detr->params()) {
          // This line is needed because of frozen batchnorms
          if(! p.isGradAvailable()) {
            continue;
          }
          p.grad() = p.grad() / scaleFactor;
          if (isBadArray(p.grad().array())) {
            FL_LOG(INFO) << "Grad has NaN values in 3, in proc: "
              << fl::getWorldRank();
            if (scaleFactor >= fl::kAmpMinimumScaleFactorValue) {
              scaleFactor = scaleFactor / 2.0f;
              FL_LOG(INFO)
                << "AMP: Scale factor decreased (grad). New value:\t"
                << scaleFactor;
              retrySample = true;
            } else {
              FL_LOG(FATAL)
                << "Minimum loss scale reached: "
                << fl::kAmpMinimumScaleFactorValue
                << " with over/underflowing gradients. Lowering the "
                << "learning rate, using gradient clipping, or "
                << "increasing the batch size can help resolve "
                << "loss explosion.";
            }
            scaleCounter = 1;
            break;
          }
        }
      }
      if (retrySample) {
        opt->zeroGrad();
        opt2->zeroGrad();
        continue;
      }
        //trainLossMeter.add(loss.array() / scaleFactor);
      } while (retrySample);


      opt->step();
      opt2->step();

      opt->zeroGrad();
      opt2->zeroGrad();
      //////////////////////////
      // Metrics
      /////////////////////////
      if (++idx % FLAGS_metric_iters == 0) {
        double total_time = timers["total"].value();
        double sample_per_second =
            (idx * FLAGS_data_batch_size * worldSize) / total_time;
        double forward_time = timers["forward"].value();
        double backward_time = timers["backward"].value();
        double criterion_time = timers["criterion"].value();
        std::stringstream ss;
        ss << "Epoch: " << epoch << std::setprecision(5) << " | Batch: " << idx
           << " | total_time: " << total_time << " | idx: " << idx
           << " | sample_per_second: " << sample_per_second
           << " | forward_time_avg: " << forward_time / idx
           << " | backward_time_avg: " << backward_time / idx
           << " | criterion_time_avg: " << criterion_time / idx;
        for (auto meter : meters) {
          fl::ext::syncMeter(meter.second);
          ss << " | " << meter.first << ": " << meter.second.value()[0];
        }
        ss << std::endl;
        FL_LOG_MASTER(INFO) << ss.str();
      }
    }
    for (auto timer : timers) {
      timer.second.reset();
    }
    for (auto meter : meters) {
      meter.second.reset();
    }
    if (fl::getWorldRank() == 0) {
      std::string filename =
          getRunFile(format("model_last.bin", idx), runIdx, runPath);
      config[kEpoch] = std::to_string(epoch);
      Serializer::save(filename, "0.1", config, detr, opt, opt2);
      filename =
          getRunFile(format("model_iter_%03d.bin", epoch), runIdx, runPath);
      Serializer::save(filename, "0.1", config, detr, opt, opt2);
    }
    if (epoch % FLAGS_eval_iters == 0) {
      evalLoop(detr, val_ds);
    }
  }
}
