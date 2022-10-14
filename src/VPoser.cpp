/* Author: Masaki Murooka */

#include <experimental/filesystem>
#include <limits>

#include <nlohmann/json.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xjson.hpp>

#include <smplpp/VPoser.h>
#include <smplpp/toolbox/Exception.h>

using namespace smplpp;

void smplpp::checkTensorNan(const torch::Tensor & tensor, const std::string & tensorName)
{
  if(at::isnan(tensor).any().item<bool>())
  {
    std::cout << tensorName << ":\n" << tensor << std::endl;
    throw smplpp::smpl_error("checkTensorNan", tensorName + " has NaN!");
  }
}

torch::Tensor smplpp::convertRotMatToAxisAngle(const torch::Tensor & rotMat)
{
  // Ref.
  // https://github.com/jrl-umi3218/SpaceVecAlg/blob/676a64c47d650ba5de6a4b0ff4f2aaf7262ffafe/src/SpaceVecAlg/PTransform.h#L293-L342

  torch::Device device = rotMat.device();

  constexpr double eps = std::numeric_limits<float>::epsilon();
  constexpr double epsSqrt = std::sqrt(eps);
  constexpr double epsSqrt2 = std::sqrt(epsSqrt);

  torch::Tensor trace = rotMat.index({at::indexing::Slice()}).diagonal(0, 1, 2).sum(-1);
  // See https://github.com/pytorch/pytorch/issues/8069
  torch::Tensor theta = at::arccos(at::clamp(0.5 * (trace - 1.0), -1.0 + eps, 1.0 - eps));

  torch::Tensor w = torch::empty({rotMat.sizes()[0], 3}, device);
  w.index_put_({at::indexing::Slice(), 0},
               rotMat.index({at::indexing::Slice(), 2, 1}) - rotMat.index({at::indexing::Slice(), 1, 2}));
  w.index_put_({at::indexing::Slice(), 1},
               rotMat.index({at::indexing::Slice(), 0, 2}) - rotMat.index({at::indexing::Slice(), 2, 0}));
  w.index_put_({at::indexing::Slice(), 2},
               rotMat.index({at::indexing::Slice(), 1, 0}) - rotMat.index({at::indexing::Slice(), 0, 1}));

  torch::Tensor aa = torch::empty_like(w, device);

  auto thetaPiCondRes = (1.0 + trace < epsSqrt2);
  torch::Tensor s = (2.0 * rotMat.index({thetaPiCondRes}).diagonal(0, 1, 2)
                     + (1.0 - trace.index({thetaPiCondRes})).view({-1, 1}).expand({-1, 3}))
                    / (3.0 - trace.index({thetaPiCondRes})).view({-1, 1});
  // Apply clamp_min to avoid NaN gradient. However, this reduces accuracy
  torch::Tensor tn2 = at::sqrt(at::clamp_min(s, eps)) * theta.index({thetaPiCondRes}).view({-1, 1});

  auto thetaPiCondRes1 = theta.index({thetaPiCondRes}) > M_PI - 1e-4;

  auto thetaPiCondRes1Y1 = tn2.index({at::indexing::Slice(), 0}).index({thetaPiCondRes1}) > 0.0;
  auto thetaPiCondRes1Y1Y1 =
      rotMat.index({thetaPiCondRes}).index({thetaPiCondRes1}).index({thetaPiCondRes1Y1, 0, 1})
          + rotMat.index({thetaPiCondRes}).index({thetaPiCondRes1}).index({thetaPiCondRes1Y1, 1, 0})
      < 0.0;
  auto thetaPiCondRes1Y1Y2 =
      rotMat.index({thetaPiCondRes}).index({thetaPiCondRes1}).index({thetaPiCondRes1Y1, 0, 2})
          + rotMat.index({thetaPiCondRes}).index({thetaPiCondRes1}).index({thetaPiCondRes1Y1, 2, 0})
      < 0.0;
  auto thetaPiCondRes1Y1N1 =
      tn2.index({at::indexing::Slice(), 1}).index({thetaPiCondRes1}).index({~thetaPiCondRes1Y1}) > 0.0;
  auto thetaPiCondRes1Y1N1Y1 = rotMat.index({thetaPiCondRes})
                                       .index({thetaPiCondRes1})
                                       .index({~thetaPiCondRes1Y1})
                                       .index({thetaPiCondRes1Y1N1, 1, 2})
                                   + rotMat.index({thetaPiCondRes})
                                         .index({thetaPiCondRes1})
                                         .index({~thetaPiCondRes1Y1})
                                         .index({thetaPiCondRes1Y1N1, 2, 1})
                               < 0.0;
  torch::Tensor tn2ThetaPiCondRes1Y = tn2.index({thetaPiCondRes1});
  torch::Tensor tn2ThetaPiCondRes1Y1Y = tn2ThetaPiCondRes1Y.index({thetaPiCondRes1Y1});
  tn2ThetaPiCondRes1Y1Y.index_put_({thetaPiCondRes1Y1Y1, 1},
                                   -1.0 * tn2ThetaPiCondRes1Y1Y.index({thetaPiCondRes1Y1Y1, 1}));
  tn2ThetaPiCondRes1Y1Y.index_put_({thetaPiCondRes1Y1Y2, 2},
                                   -1.0 * tn2ThetaPiCondRes1Y1Y.index({thetaPiCondRes1Y1Y2, 2}));
  torch::Tensor tn2ThetaPiCondRes1Y1N = tn2ThetaPiCondRes1Y.index({~thetaPiCondRes1Y1});
  torch::Tensor tn2ThetaPiCondRes1Y1N1Y = tn2ThetaPiCondRes1Y1N.index({thetaPiCondRes1Y1N1});
  tn2ThetaPiCondRes1Y1N1Y.index_put_({thetaPiCondRes1Y1N1Y1, 2},
                                     -1.0 * tn2ThetaPiCondRes1Y1N1Y.index({thetaPiCondRes1Y1N1Y1, 2}));
  tn2ThetaPiCondRes1Y1N.index_put_({thetaPiCondRes1Y1N1}, tn2ThetaPiCondRes1Y1N1Y);
  tn2ThetaPiCondRes1Y.index_put_({thetaPiCondRes1Y1}, tn2ThetaPiCondRes1Y1Y);
  tn2.index_put_({thetaPiCondRes1}, tn2ThetaPiCondRes1Y);

  auto thetaPiCondRes1N1 = w.index({thetaPiCondRes}).index({~thetaPiCondRes1}) >= 0.0;
  torch::Tensor tn2ThetaPiCondRes1N = tn2.index({~thetaPiCondRes1});
  tn2ThetaPiCondRes1N.index_put_({~thetaPiCondRes1N1}, -1.0 * tn2ThetaPiCondRes1N.index({~thetaPiCondRes1N1}));
  tn2.index_put_({~thetaPiCondRes1}, tn2ThetaPiCondRes1N);

  aa.index_put_({thetaPiCondRes}, tn2);

  auto thetaZeroCondRes = (at::abs(3.0 - trace.index({~thetaPiCondRes})) < epsSqrt);
  torch::Tensor aaThetaPiCondResN = torch::empty_like(aa.index({~thetaPiCondRes}));
  aaThetaPiCondResN.index_put_(
      {thetaZeroCondRes}, 0.5 * w.index({~thetaPiCondRes}).index({thetaZeroCondRes})
                              * (1.0 + at::pow(theta.index({~thetaPiCondRes}).index({thetaZeroCondRes}), 2) / 6.0
                                 + at::pow(theta.index({~thetaPiCondRes}).index({thetaZeroCondRes}), 4) * 7.0 / 360.0)
                                    .view({-1, 1}));
  aaThetaPiCondResN.index_put_({~thetaZeroCondRes},
                               w.index({~thetaPiCondRes}).index({~thetaZeroCondRes})
                                   * at::div(theta.index({~thetaPiCondRes}).index({~thetaZeroCondRes}),
                                             2.0 * at::sin(theta.index({~thetaPiCondRes}).index({~thetaZeroCondRes})))
                                         .view({-1, 1}));
  aa.index_put_({~thetaPiCondRes}, aaThetaPiCondResN);

  return aa;
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

  return at::stack({axis1, axis2, axis3}, -1).view({-1, 3, 3});
}

VPoserDecoderImpl::VPoserDecoderImpl()
{
  // clang-format off
  decoderNet_ = register_module(
      "decoderNet",
      torch::nn::Sequential(torch::nn::Linear(LATENT_DIM, hiddenDim_),
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
  return convertRotMatToAxisAngle(decoderNet_->forward(latent)).view({batchSize, -1, 3});
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
       && layer0WeightArr.shape(1) == LATENT_DIM))
  {
    throw smpl_error("VPoser", "invalid dimension of decoder_net.0.weight from JSON file!");
  }
  decoderNet_->at<torch::nn::LinearImpl>(0).weight.copy_(
      torch::from_blob(layer0WeightArr.data(), {hiddenDim_, LATENT_DIM}));

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
