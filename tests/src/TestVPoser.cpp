/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <smplpp/VPoser.h>

TEST(TestDdpZmp, Test1)
{
  smplpp::VPoserDecoder vposer;
  torch::Tensor vposerIn = torch::rand({1, vposer->latentDim_});
  torch::Tensor vposerOut = vposer->forward(vposerIn);

  std::cout << "vposerIn:\n" << vposerIn << std::endl;
  std::cout << "vposerOut:\n" << vposerOut << std::endl;
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
