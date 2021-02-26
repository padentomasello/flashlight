#pragma once

#include "iostream"
#include <cassert>

#include "flashlight/app/objdet/dataset/BoxUtils.h"
#include "flashlight/fl/nn/nn.h"

// TODO check layer norm dimensions

namespace fl {
namespace app {
namespace objdet {

fl::Variable transformerMultiheadAttention(
    const fl::Variable& query,
    const fl::Variable& key,
    const fl::Variable& value,
    const fl::Variable& posEmb,
    const fl::Variable& keyPaddingMask,
    const int32_t nHead,
    const double pDropout);

class MultiheadAttention : public Container {
 public:
  MultiheadAttention();
  MultiheadAttention(
      int32_t modelDim,
      int32_t headDim,
      int32_t numHeads,
      float pDropout = 0.f);

  // queries [ E, N, L ], where L is target length, N is batch size.
  // keys / values  [ E, N, S ], where S is src length, N is batch size.
  // keyPaddingMask [ S, N ]
  std::vector<Variable> forward(
      const Variable queries,
      const Variable keys,
      const Variable values,
      const Variable keyPaddingMask);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::string prettyString() const override;

 protected:
  std::shared_ptr<Linear> wq_;
  std::shared_ptr<Linear> wk_;
  std::shared_ptr<Linear> wv_;
  std::shared_ptr<Linear> wf_;
  float pDropout_;
  int32_t numHeads_;

 private:
  FL_SAVE_LOAD_WITH_BASE(
      fl::Container,
      pDropout_,
      numHeads_,
      wq_,
      wk_,
      wv_,
      wf_)
};

class TransformerBaseLayer : public Container {
 public:
  TransformerBaseLayer();
  TransformerBaseLayer(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      float pDropout);

 protected:
  std::shared_ptr<MultiheadAttention> self_attn_;
  std::shared_ptr<Linear> w1_, w2_;
  std::shared_ptr<LayerNorm> norm1_, norm2_;
  float pDropout_;

  Variable mlp(const Variable& in);

  Variable withPosEmbed(const Variable& input, const Variable& pos);

  Variable selfAttention(
      const Variable& input,
      const Variable& pos,
      const Variable& keyPaddingMask = Variable());

 private:
  FL_SAVE_LOAD_WITH_BASE(
      fl::Container,
      pDropout_,
      self_attn_,
      w1_,
      w2_,
      norm1_,
      norm2_)
};

class TransformerEncoderLayer : public TransformerBaseLayer {
 public:
  TransformerEncoderLayer();
  TransformerEncoderLayer(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      float pDropout);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(TransformerBaseLayer)
};

class TransformerDecoderLayer : public Container {
 public:
  TransformerDecoderLayer();
  TransformerDecoderLayer(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      float pDropout);

protected:

  Variable mlp(const Variable& in);

  Variable withPosEmbed(const Variable& input, const Variable& pos);
  Variable selfAttention(
      const Variable& input,
      const Variable& pos,
      const Variable& keyPaddingMask = Variable());

  std::vector<Variable> forward(const std::vector<Variable>& input);

  std::string prettyString() const;

 private:
  std::shared_ptr<MultiheadAttention> self_attn_, encoder_attn_;
  std::shared_ptr<Linear> w1_, w2_;
  std::shared_ptr<LayerNorm> norm1_, norm2_, norm3_;
  float pDropout_;
  FL_SAVE_LOAD_WITH_BASE(
      fl::Container,
      pDropout_,
      self_attn_,
      encoder_attn_,
      w1_,
      w2_,
      norm1_,
      norm2_,
      norm3_)
};

class TransformerDecoder : public Container {
 public:
  TransformerDecoder();
  TransformerDecoder(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      int32_t layers,
      float pDropout);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::string prettyString() const override;

  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

class TransformerEncoder : public Container {
 public:
  TransformerEncoder();
  TransformerEncoder(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      int32_t layers,
      float pDropout);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

class Transformer : public Container {
 public:
  Transformer();
  Transformer(
      int32_t modelDim,
      int32_t numHeads,
      int32_t numEncoderLayers,
      int32_t numDecoderLayers,
      int32_t mlpDim,
      float pDropout);

  std::vector<Variable>
  forward(Variable src, Variable mask, Variable queryEmbed, Variable posEmbed);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::string prettyString() const override;

 private:
  std::shared_ptr<TransformerEncoder> encoder_;
  std::shared_ptr<TransformerDecoder> decoder_;
  FL_SAVE_LOAD_WITH_BASE(fl::Container, encoder_, decoder_)
};

} // namespace objdet
} // namespace app
} // namespace fl
CEREAL_REGISTER_TYPE(fl::app::objdet::Transformer)
CEREAL_REGISTER_TYPE(fl::app::objdet::MultiheadAttention)
CEREAL_REGISTER_TYPE(fl::app::objdet::TransformerEncoder)
CEREAL_REGISTER_TYPE(fl::app::objdet::TransformerEncoderLayer)
CEREAL_REGISTER_TYPE(fl::app::objdet::TransformerDecoder)
CEREAL_REGISTER_TYPE(fl::app::objdet::TransformerDecoderLayer)
