/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <smplpp/toolbox/GeometryUtils.h>

TEST(TestGeometryUtils, calcTriangleVertexWeights)
{
  std::vector<torch::Tensor> vertices;
  for(int32_t i = 0; i < 3; i++)
  {
    vertices.push_back(10.0 * (torch::rand({3}) - 0.5));
  }
  torch::Tensor weights = torch::rand({3});
  weights /= weights.sum();
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

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
