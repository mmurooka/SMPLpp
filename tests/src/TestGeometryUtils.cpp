/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <smplpp/toolbox/GeometryUtils.h>
#include <smplpp/toolbox/TorchEigenUtils.hpp>

void testCalcRotMatFromNormalOnce(const Eigen::Vector3d & normal)
{
  Eigen::Matrix3d rotMat = smplpp::calcRotMatFromNormal(normal);
  Eigen::Vector3d normalRestored = rotMat.col(2);

  EXPECT_LT((normal - normalRestored).norm(), 1e-8) << "normal:\n"
                                                    << normal << std::endl
                                                    << "normalRestored:\n"
                                                    << normalRestored << std::endl;
  EXPECT_LT((rotMat * rotMat.transpose() - Eigen::Matrix3d::Identity()).norm(), 1e-8) << "rotMat:\n"
                                                                                      << rotMat << std::endl;
  EXPECT_LT((rotMat.transpose() * rotMat - Eigen::Matrix3d::Identity()).norm(), 1e-8) << "rotMat:\n"
                                                                                      << rotMat << std::endl;
  EXPECT_LT((rotMat.col(0).cross(rotMat.col(1)) - rotMat.col(2)).norm(), 1e-8) << "rotMat:\n" << rotMat << std::endl;
}

TEST(TestGeometryUtils, calcRotMatFromNormal)
{
  testCalcRotMatFromNormalOnce(Eigen::Vector3d::UnitX());
  testCalcRotMatFromNormalOnce(Eigen::Vector3d::UnitY());
  testCalcRotMatFromNormalOnce(Eigen::Vector3d::UnitZ());
  testCalcRotMatFromNormalOnce(-1 * Eigen::Vector3d::UnitX());
  testCalcRotMatFromNormalOnce(-1 * Eigen::Vector3d::UnitY());
  testCalcRotMatFromNormalOnce(-1 * Eigen::Vector3d::UnitZ());

  for(int randIdx = 0; randIdx < 1000; randIdx++)
  {
    testCalcRotMatFromNormalOnce(Eigen::Vector3d::Random().normalized());
  }
}

void testCalcTriangleVertexWeightsOnce(const std::vector<torch::Tensor> & vertices, const torch::Tensor & weights)
{
  torch::Tensor pos = torch::zeros({3});
  for(int32_t i = 0; i < 3; i++)
  {
    pos += weights.index({i}) * vertices[i];
  }

  torch::Tensor weightsRestored = smplpp::calcTriangleVertexWeights(pos, vertices);
  torch::Tensor posRestored = torch::zeros({3});
  for(int32_t i = 0; i < 3; i++)
  {
    posRestored += weightsRestored.index({i}) * vertices[i];
  }

  EXPECT_LT(std::abs(weightsRestored.sum().item<float>() - 1.0), 1e-3) << "weights:\n"
                                                                       << weights << std::endl
                                                                       << "weightsRestored:\n"
                                                                       << weightsRestored << std::endl;
  EXPECT_LT((pos - posRestored).norm().item<float>(), 1e-3) << "weights:\n"
                                                            << weights << std::endl
                                                            << "weightsRestored:\n"
                                                            << weightsRestored << std::endl
                                                            << "pos:\n"
                                                            << pos << std::endl
                                                            << "posRestored:\n"
                                                            << posRestored << std::endl;
}

TEST(TestGeometryUtils, calcTriangleVertexWeights)
{
  std::vector<torch::Tensor> weightsCornerCases;
  weightsCornerCases.push_back(smplpp::toTorchTensor<float>(Eigen::Vector3f::UnitX(), true));
  weightsCornerCases.push_back(smplpp::toTorchTensor<float>(Eigen::Vector3f::UnitY(), true));
  weightsCornerCases.push_back(smplpp::toTorchTensor<float>(Eigen::Vector3f::UnitZ(), true));
  weightsCornerCases.push_back(smplpp::toTorchTensor<float>(Eigen::Vector3f(0.0, 0.5, 0.5), true));
  weightsCornerCases.push_back(smplpp::toTorchTensor<float>(Eigen::Vector3f(0.5, 0.0, 0.5), true));
  weightsCornerCases.push_back(smplpp::toTorchTensor<float>(Eigen::Vector3f(0.5, 0.5, 0.0), true));
  weightsCornerCases.push_back(smplpp::toTorchTensor<float>(Eigen::Vector3f::Constant(1.0 / 3.0), true));

  for(int randVerticesIdx = 0; randVerticesIdx < 100; randVerticesIdx++)
  {
    std::vector<torch::Tensor> vertices;
    for(int32_t i = 0; i < 3; i++)
    {
      vertices.push_back(10.0 * (torch::rand({3}) - 0.5));
    }

    for(const torch::Tensor & weights : weightsCornerCases)
    {
      testCalcTriangleVertexWeightsOnce(vertices, weights);
    }

    for(int randWeightsIdx = 0; randWeightsIdx < 10; randWeightsIdx++)
    {
      torch::Tensor weights = torch::rand({3});
      weights /= weights.sum();

      testCalcTriangleVertexWeightsOnce(vertices, weights);
    }
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
