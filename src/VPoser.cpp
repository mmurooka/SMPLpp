/* Author: Masaki Murooka */

#include <smplpp/VPoser.h>

using namespace smplpp;

VPoserDecoderImpl::ContinousRotReprDecoderImpl::ContinousRotReprDecoderImpl()
{
  // Workaround to avoid torch error
  // See https://github.com/pytorch/pytorch/issues/35736#issuecomment-688078143
  torch::cuda::is_available();
}

torch::Tensor VPoserDecoderImpl::ContinousRotReprDecoderImpl::forward(const torch::Tensor & input)
{
  torch::Tensor reshapedInput = input.view({-1, 3, 2});
  torch::Tensor col1 = reshapedInput.index({at::indexing::Slice(), at::indexing::Slice(), 0});
  torch::Tensor col2 = reshapedInput.index({at::indexing::Slice(), at::indexing::Slice(), 1});

  torch::Tensor axis1 = torch::nn::functional::normalize(col1, torch::nn::functional::NormalizeFuncOptions().dim(1));
  torch::Tensor axis2 = torch::nn::functional::normalize(col2 - (axis1 * col2).sum(1, true) * axis1,
                                                         torch::nn::functional::NormalizeFuncOptions().dim(-1));
  torch::Tensor axis3 = at::native::cross(axis1, axis2, 1);

  return at::native::stack({axis1, axis2, axis3}, -1);
}

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
                            torch::nn::Linear(hiddenDim_, 6 * jointNum_),
                            ContinousRotReprDecoder()
                            ));
  // clang-format on

  // Workaround to avoid torch error
  // See https://github.com/pytorch/pytorch/issues/35736#issuecomment-688078143
  torch::cuda::is_available();
}

torch::Tensor VPoserDecoderImpl::forward(const torch::Tensor & latent)
{
  int64_t batchSize = latent.sizes()[0];
  return decoderNet_->forward(latent);
  // return matrot2aa(decoderNet_(latent).view({-1, 3, 3})).view({batchSize, -1, 3}); // \todo
}
