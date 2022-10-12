/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <nlohmann/json.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xjson.hpp>

#include <smplpp/VPoser.h>

#include <ros/console.h>
#include <ros/package.h>

TEST(TestVPoser, convertRotMatToAxisAngle)
{
  for(int i = 0; i < 1000; i++)
  {
    Eigen::AngleAxisd aa(Eigen::Quaterniond::UnitRandom());
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotMat = aa.toRotationMatrix().cast<float>();
    torch::Tensor tensorIn = torch::from_blob(rotMat.data(), {1, 3, 3}).clone();
    torch::Tensor tensorOut = smplpp::convertRotMatToAxisAngle(tensorIn);
    Eigen::Vector3d aaRestored(Eigen::Map<Eigen::Vector3f>(tensorOut.data_ptr<float>(), 3, 1).cast<double>());

    EXPECT_FALSE(aaRestored.array().isNaN().any());
    EXPECT_LT((aa.angle() * aa.axis() - aaRestored).norm(), 1e-3)
        << "angle: " << aa.angle() << ", axis: " << aa.axis().transpose() << std::endl
        << "aa: " << aa.angle() * aa.axis().transpose() << std::endl
        << "aaRestored: " << aaRestored.transpose() << std::endl
        << "error: " << (aa.angle() * aa.axis() - aaRestored).transpose() << std::endl;
  }
}

TEST(TestVPoser, VPoserDecoder)
{
  std::string vposerInOutPath = ros::package::getPath("smplpp") + "/tests/data/TestVPoser.json";
  ROS_INFO_STREAM("Load VPoser input and output from " << vposerInOutPath);
  std::ifstream vposerInOutFile(vposerInOutPath);
  nlohmann::json vposerInOutJsonObj = nlohmann::json::parse(vposerInOutFile);
  xt::xarray<float> arrIn;
  xt::from_json(vposerInOutJsonObj["VPoserDecoder.in"], arrIn);
  torch::Tensor tensorIn =
      torch::from_blob(arrIn.data(), {static_cast<int64_t>(arrIn.shape(0)), static_cast<int64_t>(arrIn.shape(1))});
  xt::xarray<float> arrOut;
  xt::from_json(vposerInOutJsonObj["VPoserDecoder.out"], arrOut);
  torch::Tensor tensorOutGt =
      torch::from_blob(arrOut.data(), {static_cast<int64_t>(arrOut.shape(0)), static_cast<int64_t>(arrOut.shape(1)),
                                       static_cast<int64_t>(arrOut.shape(2))});

  smplpp::VPoserDecoder vposer;
  std::string vposerParamsPath = ros::package::getPath("smplpp") + "/data/vposer_parameters.json";
  ROS_INFO_STREAM("Load VPoser parameters from " << vposerParamsPath);
  vposer->loadParamsFromJson(vposerParamsPath);
  vposer->eval();
  torch::Tensor tensorOutPred = vposer->forward(tensorIn);

  ROS_INFO_STREAM("Parameters in VPoserDecoder:");
  for(const auto & paramKV : vposer->named_parameters())
  {
    std::string key = paramKV.key();
    std::cout << "  - " << key << std::endl;
  }

  EXPECT_LT((tensorOutPred - tensorOutGt).norm().item<float>(), 1e-6) << "tensorIn:\n"
                                                                      << tensorIn << std::endl
                                                                      << "tensorOutPred:\n"
                                                                      << tensorOutPred << std::endl
                                                                      << "tensorOutGt:\n"
                                                                      << tensorOutGt << std::endl
                                                                      << "tensorOutError:\n"
                                                                      << (tensorOutPred - tensorOutGt) << std::endl;

  EXPECT_FALSE(vposer->decoderNet_->at<torch::nn::DropoutImpl>(2).is_training());
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
