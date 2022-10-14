/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <nlohmann/json.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xjson.hpp>

#include <smplpp/VPoser.h>

#include <ros/console.h>
#include <ros/package.h>

void testConvertRotMatToAxisAngleOnce(const Eigen::Matrix3d & rotMat)
{
  Eigen::AngleAxisd aa(rotMat);
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotMatFloatRowMajor = rotMat.cast<float>();
  torch::Tensor tensorIn = torch::from_blob(rotMatFloatRowMajor.data(), {1, 3, 3}).clone();
  torch::Tensor tensorOut = smplpp::convertRotMatToAxisAngle(tensorIn);
  Eigen::Vector3d aaRestored(Eigen::Map<Eigen::Vector3f>(tensorOut.data_ptr<float>(), 3, 1).cast<double>());

  EXPECT_FALSE(aaRestored.array().isNaN().any());
  EXPECT_TRUE((aa.angle() * aa.axis() - aaRestored).norm() < 1e-3
              || (std::abs(aa.angle() - M_PI) < 1e-3 && (aa.angle() * aa.axis() + aaRestored).norm() < 1e-3))
      << "angle: " << aa.angle() << ", axis: " << aa.axis().transpose() << std::endl
      << "aa: " << aa.angle() * aa.axis().transpose() << std::endl
      << "aaRestored: " << aaRestored.transpose() << std::endl
      << "error: " << (aa.angle() * aa.axis() - aaRestored).norm() << std::endl;
}

TEST(TestVPoser, convertRotMatToAxisAngle)
{
  testConvertRotMatToAxisAngleOnce(Eigen::Matrix3d::Identity());
  for(int axisIdx = 0; axisIdx < 3; axisIdx++)
  {
    for(double epsAbs : std::vector<double>{0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1})
    {
      for(int epsSign : std::vector<int>{1, -1})
      {
        double eps = epsSign * epsAbs;
        testConvertRotMatToAxisAngleOnce(Eigen::AngleAxisd(eps, Eigen::Vector3d::Unit(axisIdx)).toRotationMatrix());
        testConvertRotMatToAxisAngleOnce(
            Eigen::AngleAxisd(M_PI / 4.0 + eps, Eigen::Vector3d::Unit(axisIdx)).toRotationMatrix());
        testConvertRotMatToAxisAngleOnce(
            Eigen::AngleAxisd(M_PI / 2.0 + eps, Eigen::Vector3d::Unit(axisIdx)).toRotationMatrix());
        testConvertRotMatToAxisAngleOnce(
            Eigen::AngleAxisd(M_PI + eps, Eigen::Vector3d::Unit(axisIdx)).toRotationMatrix());
      }
    }
  }

  for(int randomIdx = 0; randomIdx < 10000; randomIdx++)
  {
    testConvertRotMatToAxisAngleOnce(Eigen::Quaterniond::UnitRandom().toRotationMatrix());
  }
}

TEST(TestVPoser, VPoserDecoder)
{
  std::string vposerDataPath = ros::package::getPath("smplpp") + "/tests/data/TestVPoser.json";
  ROS_INFO_STREAM("Load VPoser data from " << vposerDataPath);
  std::ifstream vposerDataFile(vposerDataPath);
  nlohmann::json vposerDataJsonObj = nlohmann::json::parse(vposerDataFile);
  xt::xarray<float> arrIn;
  xt::from_json(vposerDataJsonObj["VPoserDecoder.in"], arrIn);
  torch::Tensor tensorIn =
      torch::from_blob(arrIn.data(), {static_cast<int64_t>(arrIn.shape(0)), static_cast<int64_t>(arrIn.shape(1))});
  xt::xarray<float> arrOut;
  xt::from_json(vposerDataJsonObj["VPoserDecoder.out"], arrOut);
  torch::Tensor tensorOutGt =
      torch::from_blob(arrOut.data(), {static_cast<int64_t>(arrOut.shape(0)), static_cast<int64_t>(arrOut.shape(1)),
                                       static_cast<int64_t>(arrOut.shape(2))});
  xt::xarray<float> arrGrad;
  xt::from_json(vposerDataJsonObj["VPoserDecoder.grad"], arrGrad);
  torch::Tensor tensorGradGt = torch::from_blob(
      arrGrad.data(), {static_cast<int64_t>(arrGrad.shape(0)), static_cast<int64_t>(arrGrad.shape(1))});

  smplpp::VPoserDecoder vposer;
  std::string vposerParamsPath = ros::package::getPath("smplpp") + "/data/vposer_parameters.json";
  ROS_INFO_STREAM("Load VPoser parameters from " << vposerParamsPath);
  vposer->loadParamsFromJson(vposerParamsPath);
  vposer->eval();

  tensorIn.set_requires_grad(true);
  torch::Tensor tensorOutPred = vposer->forward(tensorIn);
  torch::Tensor tensorOutPredNorm = tensorOutPred.norm();
  tensorOutPredNorm.backward();
  torch::Tensor tensorGradPred = tensorIn.grad();

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

  EXPECT_LT((tensorGradPred - tensorGradGt).norm().item<float>(), 1e-6) << "tensorIn:\n"
                                                                        << tensorIn << std::endl
                                                                        << "tensorGradPred:\n"
                                                                        << tensorGradPred << std::endl
                                                                        << "tensorGradGt:\n"
                                                                        << tensorGradGt << std::endl
                                                                        << "tensorGradError:\n"
                                                                        << (tensorGradPred - tensorGradGt) << std::endl;

  EXPECT_FALSE(vposer->decoderNet_->at<torch::nn::DropoutImpl>(2).is_training());
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
