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

#define SINGLE_SMPL smplpp::Singleton<smplpp::SMPL>

//===== INCLUDES ==============================================================

//----------
#include <chrono>
//----------
#include <Eigen/Dense>
//----------
#include <torch/torch.h>
//----------
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
//----------
#include <smplpp/SMPL.h>
#include <smplpp/VPoser.h>
#include <smplpp/definition/def.h>
#include <smplpp/toolbox/Singleton.hpp>
#include <smplpp/toolbox/TorchEigenUtils.h>
#include <smplpp/toolbox/TorchEx.hpp>
//----------
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <smplpp/PoseParam.h>
//----------

//===== FORWARD DECLARATIONS ==================================================

//===== NAMESPACE =============================================================

//===== MAIN FUNCTION =========================================================

struct IkTarget
{
  int64_t vertexIdx = -1;

  torch::Tensor targetPos = torch::zeros({3});

  IkTarget(int64_t _vertexIdx, torch::Tensor _targetPos) : vertexIdx(_vertexIdx), targetPos(_targetPos) {}
};

constexpr int64_t CONFIG_DIM = smplpp::LATENT_DIM + 12;

torch::Tensor g_config = torch::zeros({CONFIG_DIM});
torch::Tensor g_theta = torch::zeros({smplpp::JOINT_NUM + 1, 3});
torch::Tensor g_beta = torch::zeros({smplpp::SHAPE_BASIS_DIM});
std::unordered_map<std::string, IkTarget> g_ikTargetList;
int64_t g_torsoVertexIdx = 3500;

void configParamCallback(const std_msgs::Float64MultiArray::ConstPtr & msg)
{
  if(msg->data.size() != CONFIG_DIM)
  {
    ROS_WARN_STREAM("Invalid config param size: " << std::to_string(msg->data.size())
                                                  << " != " << std::to_string(CONFIG_DIM));
    return;
  }

  for(int64_t i = 0; i < CONFIG_DIM; i++)
  {
    g_config.index_put_({i}, msg->data[i]);
  }
}

void poseParamCallback(const smplpp::PoseParam::ConstPtr & msg)
{
  if(msg->angles.size() != smplpp::JOINT_NUM + 1)
  {
    ROS_WARN_STREAM("Invalid pose param size: " << std::to_string(msg->angles.size())
                                                << " != " << std::to_string(smplpp::JOINT_NUM + 1));
    return;
  }

  for(int64_t i = 0; i < smplpp::JOINT_NUM + 1; i++)
  {
    g_theta.index_put_({i, 0}, msg->angles[i].x);
    g_theta.index_put_({i, 1}, msg->angles[i].y);
    g_theta.index_put_({i, 2}, msg->angles[i].z);
  }
}

void shapeParamCallback(const std_msgs::Float64MultiArray::ConstPtr & msg)
{
  if(msg->data.size() != smplpp::SHAPE_BASIS_DIM)
  {
    ROS_WARN_STREAM("Invalid shape param size: " << std::to_string(msg->data.size())
                                                 << " != " << std::to_string(smplpp::SHAPE_BASIS_DIM));
    return;
  }

  for(int64_t i = 0; i < smplpp::SHAPE_BASIS_DIM; i++)
  {
    g_beta.index_put_({i}, msg->data[i]);
  }
}

void ikTargetPoseCallback(const geometry_msgs::TransformStamped::ConstPtr & msg)
{
  g_ikTargetList.at(msg->child_frame_id).targetPos.index_put_({0}, msg->transform.translation.x);
  g_ikTargetList.at(msg->child_frame_id).targetPos.index_put_({1}, msg->transform.translation.y);
  g_ikTargetList.at(msg->child_frame_id).targetPos.index_put_({2}, msg->transform.translation.z);
}

void clickedPointCallback(const geometry_msgs::PointStamped::ConstPtr & msg)
{
  torch::Tensor clickedPos = torch::empty({3});
  clickedPos.index_put_({0}, msg->point.x);
  clickedPos.index_put_({1}, msg->point.y);
  clickedPos.index_put_({2}, msg->point.z);

  torch::Tensor vertexTensor = SINGLE_SMPL::get()->getVertex().index({0}).to(torch::kCPU);
  int64_t vertexIdx = at::argmin((vertexTensor - clickedPos.expand({smplpp::VERTEX_NUM, -1})).norm(2, 1)).item<float>();
  ROS_INFO_STREAM("Vertex idx closest to clicked point: " << vertexIdx);
}

int main(int argc, char * argv[])
{
  // Setup ROS
  ros::init(argc, argv, "smplpp");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  bool enableVposer = false;
  pnh.getParam("enable_vposer", enableVposer);
  bool enableIk = false;
  pnh.getParam("enable_ik", enableIk);
  bool enableVertexColor = false;
  pnh.getParam("enable_vertex_color", enableVertexColor);

  ros::Publisher markerArrPub = nh.advertise<visualization_msgs::MarkerArray>("smplpp/marker_arr", 1);
  ros::Publisher targetPoseArrPub = nh.advertise<geometry_msgs::PoseArray>("smplpp/target_pose_arr", 1);
  ros::Publisher actualPoseArrPub = nh.advertise<geometry_msgs::PoseArray>("smplpp/actual_pose_arr", 1);
  ros::Subscriber configParamSub;
  ros::Subscriber poseParamSub;
  if(!enableIk)
  {
    if(enableVposer)
    {
      configParamSub = nh.subscribe("smplpp/config_param", 1, configParamCallback);
    }
    else
    {
      poseParamSub = nh.subscribe("smplpp/pose_param", 1, poseParamCallback);
    }
  }
  ros::Subscriber shapeParamSub = nh.subscribe("smplpp/shape_param", 1, shapeParamCallback);
  ros::Subscriber ikTargetPoseSub = nh.subscribe("smplpp/ik_target_pose", 1, ikTargetPoseCallback);
  ros::Subscriber clickedPointSub = nh.subscribe("/clicked_point", 1, clickedPointCallback);

  // Setup IK
  if(enableIk)
  {
    // Set initial root orientation
    if(enableVposer)
    {
      g_config.index({at::indexing::Slice(3, 6)}).fill_(1.2092);
    }
    else
    {
      g_theta.index({1}).fill_(1.2092);
    }

    auto makeTensor3d = [](const std::vector<double> & vec) -> torch::Tensor {
      torch::Tensor tensor = torch::empty({3});
      for(int64_t i = 0; i < 3; i++)
      {
        tensor.index_put_({i}, vec[i]);
      }
      return tensor;
    };
    g_ikTargetList.emplace("LeftHand", IkTarget(2006, makeTensor3d({0.5, 0.5, 1.0})));
    g_ikTargetList.emplace("RightHand", IkTarget(5508, makeTensor3d({0.5, -0.5, 1.0})));
    g_ikTargetList.emplace("LeftFoot", IkTarget(3438, makeTensor3d({0.0, 0.2, 0.0})));
    g_ikTargetList.emplace("RightFoot", IkTarget(6838, makeTensor3d({0.0, -0.2, 0.0})));
  }

  // Set device
  std::unique_ptr<torch::Device> device;
  {
    std::string deviceType = "CPU";
    pnh.getParam("device", deviceType);
    if(deviceType == "CPU")
    {
      device = std::make_unique<torch::Device>(torch::kCPU);
    }
    else if(deviceType == "CUDA")
    {
      device = std::make_unique<torch::Device>(torch::kCUDA);
    }
    else
    {
      throw smplpp::smpl_error("node", "Invalid device type: " + deviceType);
    }
    device->set_index(0);
    ROS_INFO_STREAM("Device type: " << deviceType);
  }

  // Load SMPL model
  {
    auto startTime = std::chrono::system_clock::now();

    std::string smplPath;
    pnh.getParam("smpl_path", smplPath);
    if(smplPath.empty())
    {
      smplPath = ros::package::getPath("smplpp") + "/data/smpl_male.json";
    }
    ROS_INFO_STREAM("Load SMPL model from " << smplPath);

    SINGLE_SMPL::get()->setDevice(*device);
    SINGLE_SMPL::get()->setModelPath(smplPath);
    SINGLE_SMPL::get()->init();

    ROS_INFO_STREAM("Duration to load SMPL: " << std::chrono::duration_cast<std::chrono::duration<double>>(
                                                     std::chrono::system_clock::now() - startTime)
                                                     .count()
                                              << " [s]");
  }

  // Load VPoser model
  smplpp::VPoserDecoder vposer;
  if(enableVposer)
  {
    auto startTime = std::chrono::system_clock::now();

    std::string vposerPath;
    pnh.getParam("vposer_path", vposerPath);
    if(vposerPath.empty())
    {
      vposerPath = ros::package::getPath("smplpp") + "/data/vposer_parameters.json";
    }
    ROS_INFO_STREAM("Load VPoser model from " << vposerPath);

    vposer->loadParamsFromJson(vposerPath);
    vposer->eval();
    vposer->to(*device);

    ROS_INFO_STREAM("Duration to load VPoser: " << std::chrono::duration_cast<std::chrono::duration<double>>(
                                                       std::chrono::system_clock::now() - startTime)
                                                       .count()
                                                << " [s]");
  }

  double rateFreq = 30.0;
  pnh.getParam("rate", rateFreq);
  ros::Rate rate(rateFreq);
  while(ros::ok())
  {
    // Update SMPL model and solve IK
    {
      auto startTime = std::chrono::system_clock::now();

      // Setup gradient
      if(g_ikTargetList.size() > 0)
      {
        if(enableVposer)
        {
          g_config.set_requires_grad(true);
          auto & g_config_grad = g_config.mutable_grad();
          if(g_config_grad.defined())
          {
            g_config_grad = g_config_grad.detach();
            g_config_grad.zero_();
          }
        }
        else
        {
          g_theta.set_requires_grad(true);
          auto & g_theta_grad = g_theta.mutable_grad();
          if(g_theta_grad.defined())
          {
            g_theta_grad = g_theta_grad.detach();
            g_theta_grad.zero_();
          }
        }
      }

      // Forward SMPL model
      torch::Tensor theta;
      if(enableVposer)
      {
        theta = torch::zeros_like(g_theta);
        theta.index_put_({0}, g_config.index({at::indexing::Slice(0, 3)}));
        theta.index_put_({1}, g_config.index({at::indexing::Slice(3, 6)}));
        torch::Tensor vposerIn = g_config.index({at::indexing::Slice(6, smplpp::LATENT_DIM + 6)}).clone().to(*device);
        torch::Tensor vposerOut = vposer->forward(vposerIn.view({1, -1})).index({0});
        theta.index_put_({at::indexing::Slice(2, 2 + 21)}, vposerOut);
        theta.index_put_({23}, g_config.index({at::indexing::Slice(smplpp::LATENT_DIM + 6, smplpp::LATENT_DIM + 9)}));
        theta.index_put_({24}, g_config.index({at::indexing::Slice(smplpp::LATENT_DIM + 9, smplpp::LATENT_DIM + 12)}));
      }
      else
      {
        theta = g_theta;
      }
      SINGLE_SMPL::get()->launch(g_beta.view({1, -1}), theta.view({1, theta.size(0), theta.size(1)}));

      // Solve IK
      if(g_ikTargetList.size() > 0)
      {
        int64_t configDim = (enableVposer ? CONFIG_DIM : 3 * (smplpp::JOINT_NUM + 1));
        Eigen::VectorXd e(3 * g_ikTargetList.size());
        Eigen::MatrixXd J(3 * g_ikTargetList.size(), configDim);
        int64_t rowIdx = 0;

        // Add end-effector position error
        for(const auto & ikTarget : g_ikTargetList)
        {
          torch::Tensor actualPos = SINGLE_SMPL::get()->getVertexRaw(ikTarget.second.vertexIdx).to(torch::kCPU);
          torch::Tensor posError = actualPos - ikTarget.second.targetPos;

          // Set task value
          e.segment<3>(rowIdx) = smplpp::toEigenMatrix(posError).cast<double>();

          // Set task Jacobian
          for(int64_t i = 0; i < 3; i++)
          {
            // Calculate backward of each element
            torch::Tensor select = torch::zeros({1, 3});
            select.index_put_({0, i}, 1);
            actualPos.backward(select, true);

            if(enableVposer)
            {
              J.row(rowIdx) = smplpp::toEigenMatrix(g_config.grad().view({configDim})).transpose().cast<double>();
              g_config.mutable_grad().zero_();
            }
            else
            {
              J.row(rowIdx) = smplpp::toEigenMatrix(g_theta.grad().view({configDim})).transpose().cast<double>();
              g_theta.mutable_grad().zero_();
            }

            rowIdx++;
          }
        }

        // Solve linear equation for IK
        Eigen::MatrixXd linearEqA = J.transpose() * J;
        Eigen::VectorXd linearEqB = J.transpose() * e;
        {
          double deltaConfigRegWeight = 1e-3 + e.squaredNorm();
          linearEqA.diagonal().array() += deltaConfigRegWeight;
        }
        if(enableVposer)
        {
          double nominalConfigRegWeight = 1e-5;
          Eigen::VectorXd nominalConfigRegWeightVec = Eigen::VectorXd::Constant(configDim, nominalConfigRegWeight);
          nominalConfigRegWeightVec.head<6>().setZero();
          nominalConfigRegWeightVec.tail<6>().setConstant(1e3);
          linearEqA.diagonal() += nominalConfigRegWeightVec;
          linearEqB +=
              nominalConfigRegWeightVec.cwiseProduct(smplpp::toEigenMatrix(g_config.view({configDim})).cast<double>());
        }
        Eigen::LLT<Eigen::MatrixXd> linearEqLlt(linearEqA);
        if(linearEqLlt.info() == Eigen::NumericalIssue)
        {
          throw smplpp::smpl_error("node", "LLT has numerical issue!");
        }
        Eigen::VectorXd deltaConfig = -1 * linearEqLlt.solve(linearEqB);

        // Update config
        if(enableVposer)
        {
          g_config.set_requires_grad(false);
          g_config += smplpp::toTorchTensor(deltaConfig.cast<float>(), true).view_as(g_config);
        }
        else
        {
          g_theta.set_requires_grad(false);
          g_theta += smplpp::toTorchTensor(deltaConfig.cast<float>(), true).view_as(g_theta);
        }
      }

      ROS_INFO_STREAM_THROTTLE(10.0,
                               "Duration to update SMPL: " << std::chrono::duration_cast<std::chrono::duration<double>>(
                                                                  std::chrono::system_clock::now() - startTime)
                                                                      .count()
                                                                  * 1e3
                                                           << " [ms]");
    }

    // Publish SMPL model
    {
      auto startTime = std::chrono::system_clock::now();

      visualization_msgs::MarkerArray markerArrMsg;
      visualization_msgs::Marker markerMsg;
      markerMsg.header.stamp = ros::Time::now();
      markerMsg.header.frame_id = "world";
      markerMsg.ns = "SMPL model";
      markerMsg.id = 0;
      markerMsg.type = visualization_msgs::Marker::TRIANGLE_LIST;
      markerMsg.action = visualization_msgs::Marker::ADD;
      markerMsg.pose.orientation.w = 1.0;
      markerMsg.scale.x = 1.0;
      markerMsg.scale.y = 1.0;
      markerMsg.scale.z = 1.0;
      markerMsg.color.r = 0.1;
      markerMsg.color.g = 0.8;
      markerMsg.color.b = 0.1;
      markerMsg.color.a = 1.0;

      torch::Tensor vertexTensor = SINGLE_SMPL::get()->getVertex().index({0}).to(torch::kCPU);
      torch::Tensor faceIndexTensor = SINGLE_SMPL::get()->getFaceIndex().to(torch::kCPU);
      assert(vertexTensor.sizes() == torch::IntArrayRef({smplpp::VERTEX_NUM, 3}));
      assert(faceIndexTensor.sizes() == torch::IntArrayRef({smplpp::FACE_INDEX_NUM, 3}));

      xt::xarray<float> vertexArr =
          xt::adapt(vertexTensor.data_ptr<float>(),
                    xt::xarray<float>::shape_type({static_cast<const size_t>(smplpp::VERTEX_NUM), 3}));
      xt::xarray<int32_t> faceIndexArr =
          xt::adapt(faceIndexTensor.data_ptr<int32_t>(),
                    xt::xarray<int32_t>::shape_type({static_cast<const size_t>(smplpp::FACE_INDEX_NUM), 3}));

      double zMin = 0.0;
      double zMax = 0.0;
      if(enableVertexColor)
      {
        zMin = torch::min(vertexTensor.index({at::indexing::Slice(), 2})).item<float>();
        zMax = torch::max(vertexTensor.index({at::indexing::Slice(), 2})).item<float>();
      }
      auto makeColorMsg = [zMin, zMax](double z) -> std_msgs::ColorRGBA {
        std_msgs::ColorRGBA colorMsg;
        double scale = 0.5 * std::max(zMax - zMin, 1e-3);
        double zMid = 0.5 * (zMin + zMax);
        colorMsg.r = std::exp(-1 * std::pow((z - zMax) / scale, 2));
        colorMsg.g = std::exp(-1 * std::pow((z - zMid) / scale, 2));
        colorMsg.b = std::exp(-1 * std::pow((z - zMin) / scale, 2));
        colorMsg.a = 1.0;
        return colorMsg;
      };

      for(int64_t i = 0; i < smplpp::FACE_INDEX_NUM; i++)
      {
        for(int64_t j = 0; j < 3; j++)
        {
          int64_t idx = faceIndexArr(i, j) - 1;
          geometry_msgs::Point pointMsg;
          pointMsg.x = vertexArr(idx, 0);
          pointMsg.y = vertexArr(idx, 1);
          pointMsg.z = vertexArr(idx, 2);
          markerMsg.points.push_back(pointMsg);
          if(enableVertexColor)
          {
            markerMsg.colors.push_back(makeColorMsg(pointMsg.z));
          }
        }
      }

      markerArrMsg.markers.push_back(markerMsg);
      markerArrPub.publish(markerArrMsg);

      ROS_INFO_STREAM_THROTTLE(10.0, "Duration to publish message: "
                                         << std::chrono::duration_cast<std::chrono::duration<double>>(
                                                std::chrono::system_clock::now() - startTime)
                                                    .count()
                                                * 1e3
                                         << " [ms]");
    }

    // Publish IK pose
    if(g_ikTargetList.size() > 0)
    {
      geometry_msgs::PoseArray targetPoseArrMsg;
      geometry_msgs::PoseArray actualPoseArrMsg;

      auto timeNow = ros::Time::now();
      targetPoseArrMsg.header.stamp = timeNow;
      targetPoseArrMsg.header.frame_id = "world";
      actualPoseArrMsg.header.stamp = timeNow;
      actualPoseArrMsg.header.frame_id = "world";

      for(const auto & ikTarget : g_ikTargetList)
      {
        geometry_msgs::Pose targetPoseMsg;
        geometry_msgs::Pose actualPoseMsg;

        targetPoseMsg.position.x = ikTarget.second.targetPos.index({0}).item<float>();
        targetPoseMsg.position.y = ikTarget.second.targetPos.index({1}).item<float>();
        targetPoseMsg.position.z = ikTarget.second.targetPos.index({2}).item<float>();
        targetPoseMsg.orientation.w = 1.0;
        targetPoseArrMsg.poses.push_back(targetPoseMsg);

        torch::Tensor actualPos = SINGLE_SMPL::get()->getVertexRaw(ikTarget.second.vertexIdx).to(torch::kCPU);
        actualPoseMsg.position.x = actualPos.index({0}).item<float>();
        actualPoseMsg.position.y = actualPos.index({1}).item<float>();
        actualPoseMsg.position.z = actualPos.index({2}).item<float>();
        actualPoseMsg.orientation.w = 1.0;
        actualPoseArrMsg.poses.push_back(actualPoseMsg);
      }

      targetPoseArrPub.publish(targetPoseArrMsg);
      actualPoseArrPub.publish(actualPoseArrMsg);
    }

    ros::spinOnce();
    rate.sleep();
  }

  SINGLE_SMPL::destroy();

  return 0;
}

//===== CLEAN AFTERWARD =======================================================

#undef SINGLE_SMPL

//=============================================================================
