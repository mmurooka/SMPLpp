/* Author: Masaki Murooka */

#include <smplpp/VPoser.h>

using namespace smplpp;

VPoserDecoderImpl::VPoserDecoderImpl()
{
  // clang-format off
  decoderNet_ = register_module(
      "decoderNet",
      torch::nn::Sequential(torch::nn::Linear(latentDim_, hiddenDim_),
                            torch::nn::LeakyReLU(),
                            torch::nn::Dropout(0.1),
                            torch::nn::Linear(hiddenDim_, hiddenDim_),
                            torch::nn::LeakyReLU(),
                            torch::nn::Linear(hiddenDim_, 6 * jointNum_) // ,
                            // ContinousRotReprDecoder() // \todo
                            ));
  // clang-format on
}

torch::Tensor VPoserDecoderImpl::forward(const torch::Tensor & latent)
{
  int64_t batchSize = latent.sizes()[0];
  return decoderNet_->forward(latent);
  // return matrot2aa(decoderNet_(latent).view({-1, 3, 3})).view({batchSize, -1, 3}); // \todo
}
