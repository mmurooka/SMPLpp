/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <smplpp/VPoser.h>

#include <ros/console.h>
#include <ros/package.h>

TEST(TestVPoser, convertRotMatToAxisAngle)
{
  Eigen::AngleAxisd aa(0.5, Eigen::Vector3d(1.0, 2.0, 3.0).normalized());
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotMat = aa.toRotationMatrix();
  torch::Tensor tensorIn = torch::from_blob(const_cast<float *>(rotMat.cast<float>().eval().data()), {1, 3, 3}).clone();
  torch::Tensor tensorOut = smplpp::convertRotMatToAxisAngle(tensorIn);
  std::cout << "tensorIn:\n" << tensorIn << std::endl;
  std::cout << "tensorOut:\n" << tensorOut << std::endl;
  std::cout << "expected out:\n" << aa.angle() * aa.axis().transpose() << std::endl;
}

TEST(TestVPoser, VPoserDecoder)
{
  smplpp::VPoserDecoder vposer;
  std::string jsonPath = ros::package::getPath("smplpp") + "/data/vposer_parameters.json";
  ROS_INFO_STREAM("Load VPoser parameters from " << jsonPath);
  vposer->loadParamsFromJson(jsonPath);
  vposer->eval();

  // torch::Tensor vposerIn = torch::rand({1, vposer->latentDim_});
  torch::Tensor vposerIn = torch::arange(vposer->latentDim_, torch::ScalarType::Float).view({1, -1});
  torch::Tensor vposerOut = vposer->forward(vposerIn);
  // torch::Tensor vposerOut = vposer->decoderNet_->at<torch::nn::LinearImpl>(0).forward(vposerIn);

  std::cout << "vposerIn:\n" << vposerIn << std::endl;
  std::cout << "vposerOut:\n" << vposerOut << std::endl;
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
