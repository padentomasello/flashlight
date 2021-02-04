#include "flashlight/app/objdet/dataset/BoxUtils.h"
#include "flashlight/ext/image/af/Transforms.h"

#include <assert.h>

namespace {

// TODO consolidate
af::array
crop(const af::array& in, const int x, const int y, const int w, const int h) {

  assert(x + w - 1 < in.dims(0));
  assert(y + h - 1 < in.dims(1));
  return in(af::seq(x, x + w - 1), af::seq(y, y + h - 1), af::span, af::span);
}
}


namespace fl {
namespace app {
namespace objdet {

std::vector<af::array> crop(
    const std::vector<af::array>& in,
    int x,
    int y,
    int tw,
    int th
    ) {
    const af::array& image = in[0];
    const af::array croppedImage = ::crop(image, x, y, tw, th);

    const af::array& boxes = in[4];

    const std::vector<int> translateVector = { x, y, x, y };
    const std::vector<int> maxSizeVector = { tw, th };
    af::array targetSize = af::array(2, maxSizeVector.data());

    const af::array translateArray = af::array(4, translateVector.data());
    const af::array maxSizeArray = af::array(2, maxSizeVector.data());

    af::array croppedBoxes = boxes;
    af::array labels = in[5];

    if(!croppedBoxes.isempty()) {
      croppedBoxes = af::batchFunc(croppedBoxes, translateArray, af::operator-);
      croppedBoxes = af::moddims(croppedBoxes, { 2, 2, boxes.dims(1)});
      croppedBoxes = af::batchFunc(croppedBoxes, maxSizeArray, af::min); 
      croppedBoxes = af::max(croppedBoxes, 0.0);
      af::array keep = allTrue(croppedBoxes(af::span, af::seq(1, 1)) > croppedBoxes(af::span, af::seq(0, 0)));
      croppedBoxes = af::moddims(croppedBoxes, { 4, boxes.dims(1) } );
      croppedBoxes = croppedBoxes(af::span, keep);
      labels  = labels(af::span, keep);
    }
    return { croppedImage, targetSize, in[2], in[3], croppedBoxes, labels };
};

std::vector<af::array> hflip(
    const std::vector<af::array>& in) {
    af::array image = in[0];
    const int w = image.dims(0);
    const int h = image.dims(1);
    image = image(af::seq(w - 1, 0, -1), af::span, af::span, af::span);

    af::array bboxes = in[4];
    if (!bboxes.isempty()) {
      af::array bboxes_flip = af::array(bboxes.dims());
      bboxes_flip(0, af::span) = (bboxes(2, af::span) * -1) + w;
      bboxes_flip(1, af::span) = bboxes(1, af::span);
      bboxes_flip(2, af::span) = (bboxes(0, af::span) * -1) + w;
      bboxes_flip(3, af::span) = bboxes(3, af::span);
      bboxes = bboxes_flip;
    }
    return { image, in[1], in[2], in[3], bboxes, in[5]};

}

} // namespace objdet
} // namespace app
} // namespace fl
