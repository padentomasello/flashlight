#include "flashlight/app/objdet/dataset/BoxUtils.h"
#include "flashlight/app/objdet/dataset/Coco.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/app/objdet/dataset/Transforms.h"
#include "flashlight/ext/image/af/Jpeg.h"
#include "flashlight/ext/image/fl/dataset/DistributedDataset.h"
#include "flashlight/ext/image/fl/dataset/LoaderDataset.h"

#include <arrayfire.h>

#include <assert.h>
#include <algorithm>
#include <map>

namespace {

using namespace fl::app::objdet;
using namespace fl::ext::image;
using namespace fl;

using BBoxVector = std::vector<float>;
using BBoxLoader = LoaderDataset<BBoxVector>;
using FilepathLoader = LoaderDataset<std::string>;

static const int kElementsPerBbox = 4;
static const int kMaxNumLabels = 64;


std::pair<af::array, af::array> makeImageAndMaskBatch(
    const std::vector<af::array>& data
    ) {

  int maxW = -1;
  int maxH = -1;;

  for (const auto& d : data) {
    int w = d.dims(0);
    int h = d.dims(1);
    maxW = std::max(w, maxW);
    maxH = std::max(h, maxH);
  }

  af::dim4 dims = { maxW, maxH, 3, static_cast<long>(data.size()) };
  af::dim4 maskDims = { maxW, maxH, 1, static_cast<long>(data.size()) };

  auto batcharr = af::constant(0, dims);
  auto maskarr = af::constant(0, maskDims);

  for (size_t i = 0; i < data.size(); ++i) {
    af::array sample = data[i];
    af::dim4 dims = sample.dims();
    int w = dims[0];
    int h = dims[1];
    batcharr(af::seq(0, w - 1), af::seq(0, h - 1), af::span, af::seq(i, i)) = data[i];
    maskarr(af::seq(0, w - 1), af::seq(0, h - 1), af::span, af::seq(i, i)) = af::constant(1, { w, h });
  }
  return std::make_pair(batcharr, maskarr);
}

af::array makeBatch(
    const std::vector<af::array>& data
    ) {
  // Using default batching function
  if (data.empty()) {
    return af::array();
  }
  auto dims = data[0].dims();

  for (const auto& d : data) {
    if (d.dims() != dims) {
      throw std::invalid_argument("dimension mismatch while batching dataset");
    }
  }

  int ndims = (data[0].elements() > 1) ? dims.ndims() : 0;

  if (ndims >= 4) {
    throw std::invalid_argument("# of dims must be < 4 for batching");
  }
  dims[ndims] = data.size();
  auto batcharr = af::array(dims, data[0].type());

  for (size_t i = 0; i < data.size(); ++i) {
    std::array<af::seq, 4> sel{af::span, af::span, af::span, af::span};
    sel[ndims] = af::seq(i, i);
    batcharr(sel[0], sel[1], sel[2], sel[3]) = data[i];
  }
  return batcharr;
}

CocoData cocoBatchFunc(const std::vector<std::vector<af::array>>& batches) {
  // TODO padentomasello refactor
  std::vector<af::array> images(batches.size());
  std::vector<af::array> image_sizes(batches.size());
  std::vector<af::array> image_ids(batches.size());
  std::vector<af::array> original_image_sizes(batches.size());
  std::vector<af::array> target_bboxes(batches.size());
  std::vector<af::array> target_classes(batches.size());

  std::transform(batches.begin(), batches.end(), images.begin(),
      [](const std::vector<af::array>& in) { return in[ImageIdx]; }
  );
  std::transform(batches.begin(), batches.end(), image_sizes.begin(),
      [](const std::vector<af::array>& in) { return in[TargetSizeIdx]; }
  );
  std::transform(batches.begin(), batches.end(), image_ids.begin(),
      [](const std::vector<af::array>& in) { return in[ImageIdIdx]; }
  );
  std::transform(batches.begin(), batches.end(), original_image_sizes.begin(),
      [](const std::vector<af::array>& in) { return in[OriginalSizeIdx]; }
  );

  std::transform(batches.begin(), batches.end(), target_bboxes.begin(),
      [](const std::vector<af::array>& in) { return in[BboxesIdx]; }
  );
  std::transform(batches.begin(), batches.end(), target_classes.begin(),
      [](const std::vector<af::array>& in) { return in[ClassesIdx]; }
  );

  af::array imageBatch, masks;
  std::tie(imageBatch, masks) = makeImageAndMaskBatch(images);
  return {
    imageBatch,
    masks,
    makeBatch(image_sizes),
    makeBatch(image_ids),
    makeBatch(original_image_sizes),
    target_bboxes,
    target_classes
  };
}

int64_t getImageId(const std::string fp) {
    const std::string slash("/");
    const std::string period(".");
    int start = fp.rfind(slash);
    int end = fp.rfind(period);
    std::string substring = fp.substr(start + 1, end - start);
    return std::stol(substring);
}

}

namespace fl {
namespace app {
namespace objdet {

struct CocoDataSample {
  std::string filepath;
  std::vector<float> bboxes;
  std::vector<float> classes;
};

CocoDataset::CocoDataset(
    const std::string& list_file,
    std::vector<ImageTransform>& transformfns,
    int world_rank,
    int world_size,
    int batch_size,
    int num_threads,
    int prefetch_size,
    bool val
  ) {
  // Create vector of CocoDataSample which will be loaded into arrayfire arrays
  std::vector<CocoDataSample> data;
  std::ifstream ifs(list_file);
  if(!ifs) {
    throw std::runtime_error("Could not open list file: " + list_file);
  }
  // We use tabs a deliminators between the filepath and each bbox
  // We use spaced to seperate the different fields of the bbox
  const std::string delim = "\t";
  const std::string bbox_delim = " ";
  std::string line;
  while(std::getline(ifs, line)) {
      int item = line.find(delim);
      std::string filepath = line.substr(0, item);
      std::vector<float> bboxes;
      std::vector<float> classes;
      item = line.find(delim, item);
      if(item == std::string::npos) {
        data.emplace_back(CocoDataSample{ filepath, bboxes, classes });
        continue;
      }
      while(item != std::string::npos) {
        int pos = item;
        int next;
        for(int i = 0; i < 4; i++) {
          next = line.find(bbox_delim, pos + 2);
          assert(next != std::string::npos);
          bboxes.emplace_back(std::stof(line.substr(pos, next - pos)));
          pos = next;
        }
        next = line.find(bbox_delim, pos + 2);
        classes.emplace_back(std::stod(line.substr(pos, next - pos)));
        item = line.find(delim, pos);
      }
      data.emplace_back(CocoDataSample{filepath, bboxes, classes});
  }
  assert(data.size() > 0);

  // Create base dataset dataset
  auto base = std::make_shared<LoaderDataset<CocoDataSample>>(data,
    [](const CocoDataSample& sample) {
      af::array image = loadJpeg(sample.filepath);
      long long int imageSizeArray[] = { image.dims(1), image.dims(0) };
      af::array targetSize = af::array(2, imageSizeArray);
      af::array imageId = af::constant(getImageId(sample.filepath), 1, s64);

      const int num_elements = sample.bboxes.size();
      const int num_bboxes = num_elements / kElementsPerBbox;
      af::array bboxes, classes;
      if (num_bboxes > 0) {
        bboxes = af::array(kElementsPerBbox, num_bboxes, sample.bboxes.data());
        classes = af::array(1, num_bboxes, sample.classes.data());
      } else {
        bboxes = af::array(0, 1, 1, 1);
        classes = af::array(0, 1, 1, 1);
      }
      // image, size, imageId, original_size
      return std::vector<af::array>{ image, targetSize, imageId, targetSize, bboxes, classes };
  });

  std::shared_ptr<Dataset> transformed = base;

  int maxSize = 1333;
  if (val) {
    transformed = std::make_shared<TransformAllDataset>(
         transformed, randomResize({800}, maxSize));
   } else {

     std::vector<int> scales = {480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800};
     TransformAllFunction trainTransform = compose({
       randomHorizontalFlip(0.5),
       randomSelect({
         randomResize(scales, maxSize),
         compose({
            randomResize({400, 500, 600}, -1),
            randomSizeCrop(384, 600),
            randomResize(scales, 1333)
          })
       })
     });

      transformed = std::make_shared<TransformAllDataset>(
           transformed, trainTransform);
   }

  transformed = std::make_shared<TransformAllDataset>(
      transformed, Normalize);

  transformed = std::make_shared<TransformDataset>(
      transformed, transformfns);

  auto next = transformed;
  if (!val) {
    shuffled_ = std::make_shared<ShuffleDataset>(next);
    next = shuffled_;
  }
  //auto next = transformed;
  //
  auto permfn = [world_size, world_rank](int64_t idx) {
    return (idx * world_size) + world_rank;
  };
  auto sampled = std::make_shared<ResampleDataset>(
    next, permfn, next->size() / world_size);

  //auto prefetch = std::make_shared<PrefetchDataset>(sampled, num_threads, prefetch_size);
  auto prefetch = sampled;
  batched_ = std::make_shared<BatchTransformDataset<CocoData>>(
      prefetch, batch_size, BatchDatasetPolicy::SKIP_LAST, cocoBatchFunc);

}

void CocoDataset::resample() {
  if(shuffled_) {
    shuffled_->resample();
  }
}


//std::shared_ptr<Dataset> CocoDataset::getImages(
    //const std::string list_file,
    //std::vector<ImageTransform>& transformfns) {
  //const std::vector<std::string> filepaths = parseImageFilepaths(list_file);
  //auto images = cocoDataLoader(filepaths);
  //return transform(images, transformfns);
//}

//std::shared_ptr<Dataset> CocoDataset::getLabels(std::string list_file) {
    //const std::vector<BBoxVector> bboxes = parseBoundingBoxes(list_file);
    //auto bboxLabels = bboxLoader(bboxes);
    //auto classLabels = classLoader(bboxes);
    //return merge({bboxLabels, classLabels});
//}



int64_t CocoDataset::size() const {
  return batched_->size();
}

CocoData CocoDataset::get(const uint64_t idx) {
  return batched_->get(idx);
}

} // namespace objdet
} // namespace app
} // namespace fl
