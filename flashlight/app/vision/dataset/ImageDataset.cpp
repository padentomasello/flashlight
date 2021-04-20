#include "flashlight/app/vision/dataset/ImageDataset.h"

#include <arrayfire.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

#include "flashlight/fl/dataset/Dataset.h"
#include "flashlight/fl/dataset/MergeDataset.h"
#include "flashlight/fl/dataset/TransformDataset.h"

#define STB_IMAGE_IMPLEMENTATION
#include "flashlight/app/vision/dataset/stb_image.h"


namespace {

/*
 * Small generic utility class for loading data from a vector of type T into an
 * vector of arrayfire arrays
 */
template <typename T>
class Loader : public fl::Dataset {

public:
 using LoadFunc = std::function<af::array(const T&)>;

 Loader(const std::vector<T>& list, LoadFunc loadfn)
     : list_(list), loadfn_(loadfn) {}

 std::vector<af::array> get(const int64_t idx) const override {
   return {loadfn_(list_[idx])};
  }

  int64_t size() const override {
    return list_.size();
  }

  private:
  std::vector<T> list_;
  LoadFunc loadfn_;
};

float randomFloat(float a, float b) {
  float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  return a + (b - a) * r;
}

/*
 * Resizes the smallest length edge of an image to be resize while keeping
 * the aspect ratio
 */
af::array resizeSmallest(const af::array& in, const int resize) {
    const int w = in.dims(0);
    const int h = in.dims(1);
    int th, tw;
    if (h > w) {
      th = (resize * h) / w;
      tw = resize;
    } else {
      th = resize;
      tw = (resize * w) / h;
    }
    return af::resize(in, tw, th, AF_INTERP_BILINEAR);
}

af::array resize(const af::array& in, const int resize) {
  return af::resize(in, resize, resize, AF_INTERP_BILINEAR);
}

af::array crop(
    const af::array& in,
    const int x,
    const int y,
    const int w,
    const int h) {
	return in(af::seq(x, x + w - 1), af::seq(y, y + h - 1), af::span, af::span);
}

af::array centerCrop(const af::array& in, const int size) {
    const int w = in.dims(0);
    const int h = in.dims(1);
    const int cropLeft = round((static_cast<float>(w) - size) / 2.);
    const int cropTop = round((static_cast<float>(h) - size) / 2.);
    return crop(in, cropLeft, cropTop, size, size);
}

/*
 * Loads a jpeg from filepath fp. Note: It will automatically convert from any
 * numnber of channels to create an array with 3 channels
 */
af::array loadJpeg(const std::string& fp) {
	int w, h, c;
  // STB image will automatically return desired_no_channels.
  // NB: c will be the original number of channels
	int desired_no_channels = 3;
	unsigned char *img = stbi_load(fp.c_str(), &w, &h, &c, desired_no_channels);
	if (img) {
		af::array result = af::array(desired_no_channels, w, h, img);
		stbi_image_free(img);
    return af::reorder(result, 1, 2, 0);
	} else {
    throw std::invalid_argument("Could not load from filepath" + fp);
	}
}

af::array loadLabel(const uint64_t x) {
  return af::constant(x, 1, 1, 1, 1, u64);
}

}

namespace fl {


ImageDataset::ImageDataset(
    std::vector<std::string> filepaths,
    std::vector<uint64_t> labels,
    std::vector<TransformFunction>& transformfns
 ) {
  // Create image loader and apply transforms
  // TransformDataset will apply each transform in a vector to the respective af::array
  // Thus, we need to `compose` all of the transforms so are each aplied
  auto images = std::make_shared<Loader<std::string>>(filepaths, loadJpeg);
  std::vector<TransformFunction> transforms = { compose(transformfns) };
  auto transformed = std::make_shared<TransformDataset>(images, transforms);

  // Create label loader
  auto targets = std::make_shared<Loader<uint64_t>>(labels, loadLabel);

  // Merge image and labels
  ds_ = std::make_shared<MergeDataset>(MergeDataset({transformed, targets}));
}

std::vector<af::array> ImageDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  return ds_->get(idx);
}

int64_t ImageDataset::size() const {
  return ds_->size();
}

Dataset::TransformFunction ImageDataset::resizeTransform(const uint64_t resize) {
  return [resize](const af::array& in) {
    return resizeSmallest(in, resize);
  };
}

Dataset::TransformFunction ImageDataset::compose(
    std::vector<TransformFunction>& transformfns) {
  return [transformfns](const af::array& in) {
    af::array out = in;
    for(auto fn: transformfns) {
      out = fn(out);
    }
    return out;
  };
}

Dataset::TransformFunction ImageDataset::centerCropTransform(
    const int size) {
  return [size](const af::array& in) {
    return centerCrop(in, size);
  };
};

Dataset::TransformFunction ImageDataset::horizontalFlipTransform(
    const float p) {
  return [p](const af::array& in) {
    af::array out = in;
    if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) > p) {
      const uint64_t w = in.dims(0);
      out = out(af::seq(w - 1, 0, -1), af::span, af::span, af::span);
    }
    return out;
  };
};

Dataset::TransformFunction ImageDataset::randomResizeCropTransform(
    const int size,
    const float scaleLow,
    const float scaleHigh,
    const float ratioLow,
    const float ratioHigh) {
  return [=] (const af::array& in) mutable {
    const int w = in.dims(0);
    const int h = in.dims(1);
    const float area = w * h;
    for(int i = 0; i < 10; i++) {
      const float scale = randomFloat(scaleLow, scaleHigh);
      const float ratio = randomFloat(log(ratioLow), log(ratioHigh));;
      const float targetArea = scale * area;
      const float targetRatio = exp(ratio);
      const int tw = round(sqrt(targetArea * targetRatio));
      const int th = round(sqrt(targetArea / targetRatio));
      if (0 < tw && tw <= w && 0 < th && th <= h) {
        const int x = rand() % (w - tw + 1);
        const int y = rand() % (h - th + 1);
        return resize(crop(in, x, y, tw, th), size);
      }
    }
    return centerCrop(resizeSmallest(in, size), size);;
  };
}

Dataset::TransformFunction ImageDataset::randomResizeTransform(
    const int low, const int high) {
  return [low, high](const af::array& in) {
    const float scale =
      static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    const int resize = low + (high - low) *  scale;
    return resizeSmallest(in, resize);
  };
};

Dataset::TransformFunction ImageDataset::randomCropTransform(
    const int tw,
    const int th) {
  return [th, tw](const af::array& in) {
    af::array out = in;
    const uint64_t w = in.dims(0);
    const uint64_t h = in.dims(1);
    const int x = rand() % (w - tw + 1);
    const int y = rand() % (h - th + 1);
    return crop(in, x, y, tw, th);
  };
};

Dataset::TransformFunction ImageDataset::normalizeImage(
    const std::vector<float>& meanVector,
    const std::vector<float>& stdVector) {
  const af::array mean(1, 1, 3, 1, meanVector.data());
  const af::array std(1, 1, 3, 1, stdVector.data());
  return [mean, std](const af::array& in) {
    af::array out = in.as(f32) / 255.f;
    out = af::batchFunc(out, mean, af::operator-);
    out = af::batchFunc(out, std, af::operator/);
    return out;
  };
};

} // namespace fl
