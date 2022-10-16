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
#include <igl/point_mesh_squared_distance.h>
//----------

//===== FORWARD DECLARATIONS ==================================================

//===== NAMESPACE =============================================================

//===== MAIN FUNCTION =========================================================

class IkTarget
{
public:
  IkTarget(int64_t _faceIdx, torch::Tensor _targetPos) : faceIdx(_faceIdx), targetPos(_targetPos) {}

  torch::Tensor getActualPos() const
  {
    torch::Tensor faceIdxTensor = SINGLE_SMPL::get()->getFaceIndexRaw(faceIdx).to(torch::kCPU);
    torch::Tensor actualPos;
    for(int32_t i = 0; i < 3; i++)
    {
      int32_t vertexIdx = faceIdxTensor.index({i}).item<int32_t>();
      torch::Tensor vertexTensor = SINGLE_SMPL::get()->getVertexRaw(vertexIdx);
      if(i == 0)
      {
        actualPos = vertexTensor;
      }
      else
      {
        actualPos += vertexTensor;
      }
    }
    actualPos /= 3.0;
    return actualPos;
  }

public:
  int64_t faceIdx;

  torch::Tensor targetPos;
};

constexpr int64_t CONFIG_DIM = smplpp::LATENT_DIM + 12;

torch::Tensor g_config = torch::zeros({CONFIG_DIM});
torch::Tensor g_theta = torch::zeros({smplpp::JOINT_NUM + 1, 3});
torch::Tensor g_beta = torch::zeros({smplpp::SHAPE_BASIS_DIM});
std::unordered_map<std::string, IkTarget> g_ikTargetList;

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
  Eigen::Vector3d clickedPos = Eigen::Vector3d(msg->point.x, msg->point.y, msg->point.z);

  torch::Tensor vertexTensor = SINGLE_SMPL::get()->getVertex().index({0}).to(torch::kCPU);
  Eigen::MatrixX3d vertexMat = smplpp::toEigenMatrix(vertexTensor).cast<double>();
  torch::Tensor faceIdxTensor = SINGLE_SMPL::get()->getFaceIndex().to(torch::kCPU);
  Eigen::MatrixX3i faceIdxMat = smplpp::toEigenMatrix<int>(faceIdxTensor);
  faceIdxMat.array() -= 1;

  Eigen::MatrixX3d facePosMat = Eigen::MatrixX3d::Zero(faceIdxMat.rows(), 3);
  for(int64_t faceIdx = 0; faceIdx < faceIdxMat.rows(); faceIdx++)
  {
    for(int32_t i = 0; i < 3; i++)
    {
      int64_t vertexIdx = faceIdxMat(faceIdx, i);
      facePosMat.row(faceIdx) += vertexMat.row(vertexIdx);
    }
  }
  facePosMat /= 3.0;

  Eigen::Index vertexIdx;
  (vertexMat.rowwise() - clickedPos.transpose()).rowwise().squaredNorm().minCoeff(&vertexIdx);
  ROS_INFO_STREAM("Vertex idx closest to clicked point: " << vertexIdx);

  Eigen::Index faceIdx;
  (facePosMat.rowwise() - clickedPos.transpose()).rowwise().squaredNorm().minCoeff(&faceIdx);
  ROS_INFO_STREAM("Face idx closest to clicked point: " << faceIdx);
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
  ros::Publisher closestPointMarkerArrPub =
      nh.advertise<visualization_msgs::MarkerArray>("smplpp/closest_point_marker_arr", 1);
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
      for(int32_t i = 0; i < 3; i++)
      {
        tensor.index_put_({i}, vec[i]);
      }
      return tensor;
    };
    g_ikTargetList.emplace("LeftHand", IkTarget(2581, makeTensor3d({0.5, 0.5, 1.0})));
    g_ikTargetList.emplace("RightHand", IkTarget(9472, makeTensor3d({0.5, -0.5, 1.0})));
    g_ikTargetList.emplace("LeftFoot", IkTarget(5925, makeTensor3d({0.0, 0.2, 0.0})));
    g_ikTargetList.emplace("RightFoot", IkTarget(12812, makeTensor3d({0.0, -0.2, 0.0})));
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
      std::vector<std::pair<std::string, double>> durationList;
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
      auto startTimeForward = std::chrono::system_clock::now();
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
      durationList.emplace_back("forward SMPL", std::chrono::duration_cast<std::chrono::duration<double>>(
                                                    std::chrono::system_clock::now() - startTimeForward)
                                                        .count()
                                                    * 1e3);

      // Solve IK
      if(g_ikTargetList.size() > 0)
      {
        int64_t configDim = (enableVposer ? CONFIG_DIM : 3 * (smplpp::JOINT_NUM + 1));
        Eigen::VectorXd e(3 * g_ikTargetList.size());
        Eigen::MatrixXd J(3 * g_ikTargetList.size(), configDim);
        int64_t rowIdx = 0;

        // Set end-effector position task
        auto startTimeSetupIk = std::chrono::system_clock::now();
        for(const auto & ikTarget : g_ikTargetList)
        {
          torch::Tensor actualPos = ikTarget.second.getActualPos();
          torch::Tensor posError = actualPos - ikTarget.second.targetPos;

          // Set task value
          e.segment<3>(rowIdx) = smplpp::toEigenMatrix(posError.to(torch::kCPU)).cast<double>();

          // Set task Jacobian
          for(int32_t i = 0; i < 3; i++)
          {
            // Calculate backward of each element
            torch::Tensor select = torch::zeros({1, 3});
            select.index_put_({0, i}, 1);
            posError.backward(select, true);

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
        durationList.emplace_back("setup IK matrices", std::chrono::duration_cast<std::chrono::duration<double>>(
                                                           std::chrono::system_clock::now() - startTimeSetupIk)
                                                               .count()
                                                           * 1e3);

        // Solve linear equation for IK
        auto startTimeSolveIk = std::chrono::system_clock::now();
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
        durationList.emplace_back("solve IK", std::chrono::duration_cast<std::chrono::duration<double>>(
                                                  std::chrono::system_clock::now() - startTimeSolveIk)
                                                      .count()
                                                  * 1e3);

        // Update config
        if(enableVposer)
        {
          g_config.set_requires_grad(false);
          g_config += smplpp::toTorchTensor<float>(deltaConfig.cast<float>(), true).view_as(g_config);
        }
        else
        {
          g_theta.set_requires_grad(false);
          g_theta += smplpp::toTorchTensor<float>(deltaConfig.cast<float>(), true).view_as(g_theta);
        }
      }

      ROS_INFO_STREAM_THROTTLE(10.0,
                               "Duration to update SMPL: " << std::chrono::duration_cast<std::chrono::duration<double>>(
                                                                  std::chrono::system_clock::now() - startTime)
                                                                      .count()
                                                                  * 1e3
                                                           << " [ms]");
      std::string durationListStr;
      for(const auto & durationKV : durationList)
      {
        durationListStr += "  - Duration to " + durationKV.first + ": " + std::to_string(durationKV.second) + " [ms]\n";
      }
      durationListStr.pop_back();
      ROS_INFO_STREAM_THROTTLE(10.0, durationListStr);
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
      Eigen::MatrixX3d vertexMat = smplpp::toEigenMatrix(vertexTensor).cast<double>();
      torch::Tensor faceIdxTensor = SINGLE_SMPL::get()->getFaceIndex().to(torch::kCPU);
      Eigen::MatrixX3i faceIdxMat = smplpp::toEigenMatrix<int>(faceIdxTensor);
      faceIdxMat.array() -= 1;

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
        for(int32_t j = 0; j < 3; j++)
        {
          int64_t idx = faceIdxMat(i, j);
          geometry_msgs::Point pointMsg;
          pointMsg.x = vertexMat(idx, 0);
          pointMsg.y = vertexMat(idx, 1);
          pointMsg.z = vertexMat(idx, 2);
          markerMsg.points.push_back(pointMsg);
          if(enableVertexColor)
          {
            markerMsg.colors.push_back(makeColorMsg(pointMsg.z));
          }
        }
      }

      markerArrMsg.markers.push_back(markerMsg);
      markerArrPub.publish(markerArrMsg);

      ROS_INFO_STREAM_THROTTLE(10.0, "Duration to publish SMPL model: "
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

        torch::Tensor actualPos = ikTarget.second.getActualPos();
        actualPoseMsg.position.x = actualPos.index({0}).item<float>();
        actualPoseMsg.position.y = actualPos.index({1}).item<float>();
        actualPoseMsg.position.z = actualPos.index({2}).item<float>();
        actualPoseMsg.orientation.w = 1.0;
        actualPoseArrMsg.poses.push_back(actualPoseMsg);
      }

      targetPoseArrPub.publish(targetPoseArrMsg);
      actualPoseArrPub.publish(actualPoseArrMsg);
    }

    // Calculate closest point
    {
      auto startTime = std::chrono::system_clock::now();

      torch::Tensor vertexTensor = SINGLE_SMPL::get()->getVertex().index({0}).to(torch::kCPU);
      Eigen::MatrixXd vertexMat = smplpp::toEigenMatrix(vertexTensor).cast<double>();
      torch::Tensor faceIdxTensor = SINGLE_SMPL::get()->getFaceIndex().to(torch::kCPU);
      Eigen::MatrixXi faceIdxMat = smplpp::toEigenMatrix<int>(faceIdxTensor);
      faceIdxMat.array() -= 1;

      Eigen::MatrixXd targetPoint = Eigen::MatrixXd::Zero(1, 3);
      Eigen::VectorXd squaredDists;
      Eigen::VectorXi closestFaceIndices;
      Eigen::MatrixX3d closestPoints;
      igl::point_mesh_squared_distance(targetPoint, vertexMat, faceIdxMat, squaredDists, closestFaceIndices,
                                       closestPoints);

      ROS_INFO_STREAM_THROTTLE(10.0, "Duration to calculate closest point: "
                                         << std::chrono::duration_cast<std::chrono::duration<double>>(
                                                std::chrono::system_clock::now() - startTime)
                                                    .count()
                                                * 1e3
                                         << " [ms]");

      visualization_msgs::MarkerArray markerArrMsg;
      auto timeNow = ros::Time::now();

      visualization_msgs::Marker sphereMarkerMsg;
      sphereMarkerMsg.header.stamp = timeNow;
      sphereMarkerMsg.header.frame_id = "world";
      sphereMarkerMsg.ns = "Closest point";
      sphereMarkerMsg.id = 0;
      sphereMarkerMsg.type = visualization_msgs::Marker::SPHERE_LIST;
      sphereMarkerMsg.action = visualization_msgs::Marker::ADD;
      sphereMarkerMsg.pose.orientation.w = 1.0;
      sphereMarkerMsg.scale.x = 0.05;
      sphereMarkerMsg.scale.y = 0.05;
      sphereMarkerMsg.scale.z = 0.05;
      sphereMarkerMsg.color.r = 0.8;
      sphereMarkerMsg.color.g = 0.1;
      sphereMarkerMsg.color.b = 0.1;
      sphereMarkerMsg.color.a = 1.0;
      sphereMarkerMsg.points.resize(2);
      sphereMarkerMsg.points[0].x = 0.0;
      sphereMarkerMsg.points[0].y = 0.0;
      sphereMarkerMsg.points[0].z = 0.0;
      sphereMarkerMsg.points[1].x = closestPoints(0, 0);
      sphereMarkerMsg.points[1].y = closestPoints(0, 1);
      sphereMarkerMsg.points[1].z = closestPoints(0, 2);

      visualization_msgs::Marker lineMarkerMsg;
      lineMarkerMsg.header.stamp = timeNow;
      lineMarkerMsg.header.frame_id = "world";
      lineMarkerMsg.ns = "Closest point";
      lineMarkerMsg.id = 1;
      lineMarkerMsg.type = visualization_msgs::Marker::LINE_LIST;
      lineMarkerMsg.action = visualization_msgs::Marker::ADD;
      lineMarkerMsg.pose.orientation.w = 1.0;
      lineMarkerMsg.scale.x = 0.01;
      lineMarkerMsg.scale.y = 0.01;
      lineMarkerMsg.scale.z = 0.01;
      lineMarkerMsg.color.r = 1.0;
      lineMarkerMsg.color.g = 0.1;
      lineMarkerMsg.color.b = 0.1;
      lineMarkerMsg.color.a = 1.0;
      lineMarkerMsg.points.resize(2);
      lineMarkerMsg.points[0].x = 0.0;
      lineMarkerMsg.points[0].y = 0.0;
      lineMarkerMsg.points[0].z = 0.0;
      lineMarkerMsg.points[1].x = closestPoints(0, 0);
      lineMarkerMsg.points[1].y = closestPoints(0, 1);
      lineMarkerMsg.points[1].z = closestPoints(0, 2);

      markerArrMsg.markers.push_back(sphereMarkerMsg);
      markerArrMsg.markers.push_back(lineMarkerMsg);
      closestPointMarkerArrPub.publish(markerArrMsg);
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
