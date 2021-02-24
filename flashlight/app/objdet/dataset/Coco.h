#pragma once
#include <gflags/gflags.h>

#include "flashlight/app/objdet/dataset/BatchTransformDataset.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/ext/image/fl/dataset/Jpeg.h"
#include "flashlight/fl/dataset/datasets.h"

namespace fl {
namespace app {
namespace objdet {

struct CocoData {
  af::array images;
  af::array masks;
  af::array imageSizes;
  af::array imageIds;
  af::array originalImageSizes;
  std::vector<af::array> target_boxes;
  std::vector<af::array> target_labels;
};

class CocoDataset {
 public:
  CocoDataset(
      const std::string& list_file,
      int world_rank,
      int world_size,
      int batch_size,
      int num_threads,
      int prefetch_size,
      bool val);

  std::shared_ptr<Dataset> getLabels(std::string list_file);

  std::shared_ptr<Dataset> getImages(
      std::string list_file,
      std::vector<ext::image::ImageTransform>& transformfns);

  using iterator = detail::DatasetIterator<CocoDataset, CocoData>;

  iterator begin() {
    return iterator(this);
  }

  iterator end() {
    return iterator();
  }

  int64_t size() const;

  CocoData get(const uint64_t idx);

  void resample();

 private:
  std::shared_ptr<BatchTransformDataset<CocoData>> batched_;
  std::shared_ptr<ShuffleDataset> shuffled_;
};

} // namespace objdet
} // namespace app
} // namespace fl
