/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <Eigen/Dense>

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
