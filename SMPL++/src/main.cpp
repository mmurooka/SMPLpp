/* ========================================================================= *
 *                                                                           *
 *                                 SMPL++                                    *
 *                    Copyright (c) 2018, Chongyi Zheng.                     *
 *                          All Rights reserved.                             *
 *                                                                           *
 * ------------------------------------------------------------------------- *
 *                                                                           *
 * This software implements a 3D human skinning model - SMPL: A Skinned      *
 * Multi-Person Linear Model with C++.                                       *
 *                                                                           *
 * For more detail, see the paper published by Max Planck Institute for      *
 * Intelligent Systems on SIGGRAPH ASIA 2015.                                *
 *                                                                           *
 * We provide this software for research purposes only.                      *
 * The original SMPL model is available at http://smpl.is.tue.mpg.           *
 *                                                                           *
 * ========================================================================= */

//=============================================================================
//
//  APPLICATION ENTRANCE
//
//=============================================================================

//===== MACROS ================================================================

#define SINGLE_SMPL smpl::Singleton<smpl::SMPL>

//===== INCLUDES ==============================================================

//----------
#include <chrono>
//----------
#include <torch/torch.h>
//----------
#include "definition/def.h"
#include "smpl/SMPL.h"
#include "toolbox/Singleton.hpp"
//----------
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
//----------

//===== FORWARD DECLARATIONS ==================================================

//===== NAMESPACE =============================================================

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

//===== MAIN FUNCTION =========================================================

int main(int argc, char * argv[])
{
  ros::init(argc, argv, "smplpp");
  ros::NodeHandle nh;
  ros::Publisher marker_pub = nh.advertise<visualization_msgs::MarkerArray>("smplpp/marker_arr", 1);

  try
  {
    torch::Device device(torch::kCPU);
    std::string modelPath = "../data/smpl_female.json";
    nh.getParam("model_path", modelPath);

    SINGLE_SMPL::get()->setDevice(device);
    SINGLE_SMPL::get()->setModelPath(modelPath);

    auto begin = clk::now();
    SINGLE_SMPL::get()->init();
    auto end = clk::now();
    auto duration = std::chrono::duration_cast<ms>(end - begin);
    ROS_INFO_STREAM("Time duration to load SMPL: " << (double)duration.count() / 1000 << " [s]");
  }
  catch(std::exception & e)
  {
    std::cerr << e.what() << std::endl;
  }

  while(ros::ok())
  {
    torch::Tensor beta = 0.03 * torch::rand({BATCH_SIZE, SHAPE_BASIS_DIM}); // (N, 10)
    torch::Tensor theta = 0.2 * torch::rand({BATCH_SIZE, JOINT_NUM, 3}); // (N, 24, 3)

    {
      auto begin = clk::now();
      SINGLE_SMPL::get()->launch(beta, theta);
      auto end = clk::now();
      auto duration = std::chrono::duration_cast<ms>(end - begin);
      ROS_INFO_STREAM("Time duration to run SMPL: " << (double)duration.count() / 1000 << " [s]");
    }

    {
      torch::Tensor vertices = SINGLE_SMPL::get()->getVertex();
    }
  }

  SINGLE_SMPL::destroy();

  return 0;
}

//===== CLEAN AFTERWARD =======================================================

#undef SINGLE_SMPL

//=============================================================================
