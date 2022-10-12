/* Author: Masaki Murooka */

#include <experimental/filesystem>

#include <nlohmann/json.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xjson.hpp>

#include <smplpp/VPoser.h>
#include <smplpp/toolbox/Exception.h>

using namespace smplpp;

torch::Tensor smplpp::convertRotMatToAxisAngle(const torch::Tensor & rotMat)
{
  // \todo Support corner cases (zero angle, 180 degrees angle)
  torch::Tensor trace = rotMat.index({at::indexing::Slice(), 0, 0}) + rotMat.index({at::indexing::Slice(), 1, 1})
                        + rotMat.index({at::indexing::Slice(), 2, 2});
  torch::Tensor angle = at::arccos((trace - 1.0) / 2.0);
  torch::Tensor axis = torch::empty({rotMat.sizes()[0], 3});
  axis.index_put_({at::indexing::Slice(), 0},
                  rotMat.index({at::indexing::Slice(), 2, 1}) - rotMat.index({at::indexing::Slice(), 1, 2}));
  axis.index_put_({at::indexing::Slice(), 1},
                  rotMat.index({at::indexing::Slice(), 0, 2}) - rotMat.index({at::indexing::Slice(), 2, 0}));
  axis.index_put_({at::indexing::Slice(), 2},
                  rotMat.index({at::indexing::Slice(), 1, 0}) - rotMat.index({at::indexing::Slice(), 0, 1}));
  torch::Tensor denom = 2.0 * at::sin(angle);
  axis.index_put_({at::indexing::Slice(), 0}, axis.index({at::indexing::Slice(), 0}));
  axis.index_put_({at::indexing::Slice(), 1}, axis.index({at::indexing::Slice(), 1}));
  axis.index_put_({at::indexing::Slice(), 2}, axis.index({at::indexing::Slice(), 2}));
  axis /= denom.view({-1, 1});
  return angle.view({-1, 1}) * axis;
}

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
  torch::Tensor axis3 = at::cross(axis1, axis2, 1);

  return at::stack({axis1, axis2, axis3}, -1);
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
  return convertRotMatToAxisAngle(decoderNet_->forward(latent).view({-1, 3, 3})).view({batchSize, -1, 3});
}

void VPoserDecoderImpl::loadParamsFromJson(const std::string & jsonPath)
{
  torch::NoGradGuard noGrad;

  nlohmann::json jsonObj;
  std::experimental::filesystem::path jsonFsPath(jsonPath);
  if(std::experimental::filesystem::exists(jsonFsPath))
  {
    std::ifstream jsonFile(jsonFsPath);
    jsonFile >> jsonObj;
  }
  else
  {
    throw smpl_error("VPoser", "Cannot find a JSON file!");
  }

  xt::xarray<float> layer0WeightArr;
  xt::from_json(jsonObj["decoder_net.0.weight"], layer0WeightArr);
  if(!(layer0WeightArr.dimension() == 2 && layer0WeightArr.shape(0) == hiddenDim_
       && layer0WeightArr.shape(1) == latentDim_))
  {
    throw smpl_error("VPoser", "invalid dimension of decoder_net.0.weight from JSON file!");
  }
  decoderNet_->at<torch::nn::LinearImpl>(0).weight.copy_(
      torch::from_blob(layer0WeightArr.data(), {hiddenDim_, latentDim_}));

  xt::xarray<float> layer0BiasArr;
  xt::from_json(jsonObj["decoder_net.0.bias"], layer0BiasArr);
  if(!(layer0BiasArr.dimension() == 1 && layer0BiasArr.shape(0) == hiddenDim_))
  {
    throw smpl_error("VPoser", "invalid dimension of decoder_net.0.bias from JSON file!");
  }
  decoderNet_->at<torch::nn::LinearImpl>(0).bias.copy_(torch::from_blob(layer0BiasArr.data(), {hiddenDim_}));

  xt::xarray<float> layer3WeightArr;
  xt::from_json(jsonObj["decoder_net.3.weight"], layer3WeightArr);
  if(!(layer3WeightArr.dimension() == 2 && layer3WeightArr.shape(0) == hiddenDim_
       && layer3WeightArr.shape(1) == hiddenDim_))
  {
    throw smpl_error("VPoser", "invalid dimension of decoder_net.3.weight from JSON file!");
  }
  decoderNet_->at<torch::nn::LinearImpl>(3).weight.copy_(
      torch::from_blob(layer3WeightArr.data(), {hiddenDim_, hiddenDim_}));

  xt::xarray<float> layer3BiasArr;
  xt::from_json(jsonObj["decoder_net.3.bias"], layer3BiasArr);
  if(!(layer3BiasArr.dimension() == 1 && layer3BiasArr.shape(0) == hiddenDim_))
  {
    throw smpl_error("VPoser", "invalid dimension of decoder_net.3.bias from JSON file!");
  }
  decoderNet_->at<torch::nn::LinearImpl>(3).bias.copy_(torch::from_blob(layer3BiasArr.data(), {hiddenDim_}));

  xt::xarray<float> layer5WeightArr;
  xt::from_json(jsonObj["decoder_net.5.weight"], layer5WeightArr);
  if(!(layer5WeightArr.dimension() == 2 && layer5WeightArr.shape(0) == 6 * jointNum_
       && layer5WeightArr.shape(1) == hiddenDim_))
  {
    throw smpl_error("VPoser", "invalid dimension of decoder_net.5.weight from JSON file!");
  }
  decoderNet_->at<torch::nn::LinearImpl>(5).weight.copy_(
      torch::from_blob(layer5WeightArr.data(), {6 * jointNum_, hiddenDim_}));

  xt::xarray<float> layer5BiasArr;
  xt::from_json(jsonObj["decoder_net.5.bias"], layer5BiasArr);
  if(!(layer5BiasArr.dimension() == 1 && layer5BiasArr.shape(0) == 6 * jointNum_))
  {
    throw smpl_error("VPoser", "invalid dimension of decoder_net.5.bias from JSON file!");
  }
  decoderNet_->at<torch::nn::LinearImpl>(5).bias.copy_(torch::from_blob(layer5BiasArr.data(), {6 * jointNum_}));
}
