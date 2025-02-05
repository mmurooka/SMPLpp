/* Author: Masaki Murooka */

#include <chrono>
#include <fstream>

#include <Eigen/Dense>

#include <torch/torch.h>

#include <smplpp/IkTask.h>
#include <smplpp/SMPL.h>
#include <smplpp/VPoser.h>
#include <smplpp/definition/def.h>
#include <smplpp/toolbox/GeometryUtils.h>
#include <smplpp/toolbox/GridUtils.hpp>
#include <smplpp/toolbox/TorchEigenUtils.hpp>
#include <smplpp/toolbox/TorchEx.hpp>

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Float64MultiArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <smplpp/Motion.h>
#include <smplpp/PoseParam.h>

#include <igl/point_mesh_squared_distance.h>
#include <igl/winding_number.h>

#include <qp_solver_collection/QpSolverCollection.h>

#include <ezc3d/Data.h>
#include <ezc3d/Header.h>
#include <ezc3d/Parameters.h>
#include <ezc3d/ezc3d.h>

constexpr int64_t LATENT_POSE_DIM = smplpp::LATENT_DIM + 12;

std::shared_ptr<smplpp::SMPL> g_smpl;
torch::Tensor g_theta;
torch::Tensor g_beta;
std::map<std::string, smplpp::IkTask> g_ikTaskList;
std::unordered_map<Eigen::Vector3i, double> g_sweepGridList;
ros::Publisher g_clickedMarkerArrPub;

void latentPoseParamCallback(const std_msgs::Float64MultiArray::ConstPtr & msg)
{
  if(msg->data.size() != LATENT_POSE_DIM)
  {
    ROS_WARN_STREAM("Invalid latent pose param size: " << std::to_string(msg->data.size())
                                                       << " != " << std::to_string(LATENT_POSE_DIM));
    return;
  }

  for(int64_t i = 0; i < LATENT_POSE_DIM; i++)
  {
    g_theta.index_put_({i}, msg->data[i]);
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
  if(g_ikTaskList.count(msg->child_frame_id) == 0)
  {
    return;
  }

  auto & ikTask = g_ikTaskList.at(msg->child_frame_id);
  ikTask.targetPos_.index_put_({0}, msg->transform.translation.x);
  ikTask.targetPos_.index_put_({1}, msg->transform.translation.y);
  ikTask.targetPos_.index_put_({2}, msg->transform.translation.z);
  Eigen::Quaterniond quat(msg->transform.rotation.w, msg->transform.rotation.x, msg->transform.rotation.y,
                          msg->transform.rotation.z);
  Eigen::Vector3d normal = quat.toRotationMatrix().col(2);
  ikTask.targetNormal_.index_put_({0}, normal.x());
  ikTask.targetNormal_.index_put_({1}, normal.y());
  ikTask.targetNormal_.index_put_({2}, normal.z());
}

void clickedPointCallback(const geometry_msgs::PointStamped::ConstPtr & msg)
{
  Eigen::Vector3f clickedPos = Eigen::Vector3f(msg->point.x, msg->point.y, msg->point.z);

  torch::Tensor vertexTensor = g_smpl->getVertex().index({0}).to(torch::kCPU);
  Eigen::MatrixX3f vertexMat = smplpp::toEigenMatrix(vertexTensor);
  torch::Tensor faceIdxTensor = g_smpl->getFaceIndex().to(torch::kCPU) - 1;
  Eigen::MatrixX3i faceIdxMat = smplpp::toEigenMatrix<int>(faceIdxTensor);

  Eigen::MatrixX3f facePosMat = Eigen::MatrixX3f::Zero(faceIdxMat.rows(), 3);
  for(int64_t faceIdx = 0; faceIdx < faceIdxMat.rows(); faceIdx++)
  {
    for(int32_t i = 0; i < 3; i++)
    {
      int32_t vertexIdx = faceIdxMat(faceIdx, i);
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

  // Publish clicked point and face
  {
    visualization_msgs::MarkerArray markerArrMsg;

    visualization_msgs::Marker pointMarkerMsg;
    pointMarkerMsg.header.stamp = ros::Time::now();
    pointMarkerMsg.header.frame_id = "world";
    pointMarkerMsg.ns = "Clicked point";
    pointMarkerMsg.id = 0;
    pointMarkerMsg.type = visualization_msgs::Marker::SPHERE;
    pointMarkerMsg.action = visualization_msgs::Marker::ADD;
    pointMarkerMsg.pose.position.x = vertexMat.row(vertexIdx).x();
    pointMarkerMsg.pose.position.y = vertexMat.row(vertexIdx).y();
    pointMarkerMsg.pose.position.z = vertexMat.row(vertexIdx).z();
    pointMarkerMsg.pose.orientation.w = 1.0;
    pointMarkerMsg.scale.x = 0.01;
    pointMarkerMsg.scale.y = 0.01;
    pointMarkerMsg.scale.z = 0.01;
    pointMarkerMsg.color.r = 0.5;
    pointMarkerMsg.color.g = 0.0;
    pointMarkerMsg.color.b = 0.0;
    pointMarkerMsg.color.a = 1.0;
    markerArrMsg.markers.push_back(pointMarkerMsg);

    visualization_msgs::Marker faceMarkerMsg;
    faceMarkerMsg.header = pointMarkerMsg.header;
    faceMarkerMsg.ns = "Clicked face";
    faceMarkerMsg.id = 1;
    faceMarkerMsg.type = visualization_msgs::Marker::LINE_LIST;
    faceMarkerMsg.action = visualization_msgs::Marker::ADD;
    faceMarkerMsg.pose.orientation.w = 1.0;
    faceMarkerMsg.scale.x = 0.005;
    faceMarkerMsg.scale.y = 0.005;
    faceMarkerMsg.scale.z = 0.005;
    faceMarkerMsg.color.r = 0.0;
    faceMarkerMsg.color.g = 0.5;
    faceMarkerMsg.color.b = 0.0;
    faceMarkerMsg.color.a = 1.0;
    torch::Tensor faceVertexIdxs = g_smpl->getFaceIndexRaw(faceIdx).to(torch::kCPU) - 1;
    // Clone and detach vertex tensors to separate tangents from the computation graph
    torch::Tensor faceVertices =
        g_smpl->getVertexRaw(faceVertexIdxs.to(torch::kInt64)).to(torch::kCPU).clone().detach();
    for(int32_t i = 0; i < 3; i++)
    {
      for(int32_t j = 0; j < 2; j++)
      {
        geometry_msgs::Point pointMsg;
        pointMsg.x = faceVertices.index({(i + j) % 3, 0}).item<float>();
        pointMsg.y = faceVertices.index({(i + j) % 3, 1}).item<float>();
        pointMsg.z = faceVertices.index({(i + j) % 3, 2}).item<float>();
        faceMarkerMsg.points.push_back(pointMsg);
      }
    }
    markerArrMsg.markers.push_back(faceMarkerMsg);

    visualization_msgs::Marker adjacentMarkerMsg;
    adjacentMarkerMsg.header = pointMarkerMsg.header;
    adjacentMarkerMsg.ns = "Clicked adjacent faces";
    adjacentMarkerMsg.id = 2;
    adjacentMarkerMsg.type = visualization_msgs::Marker::LINE_LIST;
    adjacentMarkerMsg.action = visualization_msgs::Marker::ADD;
    adjacentMarkerMsg.pose.orientation.w = 1.0;
    adjacentMarkerMsg.scale.x = 0.004;
    adjacentMarkerMsg.scale.y = 0.004;
    adjacentMarkerMsg.scale.z = 0.004;
    adjacentMarkerMsg.color.r = 0.0;
    adjacentMarkerMsg.color.g = 0.0;
    adjacentMarkerMsg.color.b = 0.5;
    adjacentMarkerMsg.color.a = 1.0;
    for(const auto & adjacentFaceKV : g_smpl->getAdjacentFaces(vertexIdx))
    {
      int64_t adjacentFaceIdx = adjacentFaceKV.first;
      torch::Tensor faceVertexIdxs = g_smpl->getFaceIndexRaw(adjacentFaceIdx).to(torch::kCPU) - 1;
      // Clone and detach vertex tensors to separate tangents from the computation graph
      torch::Tensor faceVertices =
          g_smpl->getVertexRaw(faceVertexIdxs.to(torch::kInt64)).to(torch::kCPU).clone().detach();
      for(int32_t i = 0; i < 3; i++)
      {
        for(int32_t j = 0; j < 2; j++)
        {
          geometry_msgs::Point pointMsg;
          pointMsg.x = faceVertices.index({(i + j) % 3, 0}).item<float>();
          pointMsg.y = faceVertices.index({(i + j) % 3, 1}).item<float>();
          pointMsg.z = faceVertices.index({(i + j) % 3, 2}).item<float>();
          adjacentMarkerMsg.points.push_back(pointMsg);
        }
      }
    }
    markerArrMsg.markers.push_back(adjacentMarkerMsg);

    visualization_msgs::Marker normalMarkerMsg;
    normalMarkerMsg.header = pointMarkerMsg.header;
    normalMarkerMsg.ns = "Clicked normals";
    normalMarkerMsg.id = 3;
    normalMarkerMsg.type = visualization_msgs::Marker::LINE_LIST;
    normalMarkerMsg.action = visualization_msgs::Marker::ADD;
    normalMarkerMsg.pose.orientation.w = 1.0;
    normalMarkerMsg.scale.x = 0.001;
    normalMarkerMsg.scale.y = 0.001;
    normalMarkerMsg.scale.z = 0.001;
    normalMarkerMsg.color.r = 0.5;
    normalMarkerMsg.color.g = 0.0;
    normalMarkerMsg.color.b = 0.5;
    normalMarkerMsg.color.a = 1.0;
    smplpp::IkTask ikTask(g_smpl, 0);
    for(const auto & adjacentFaceKV : g_smpl->getAdjacentFaces(vertexIdx))
    {
      ikTask.faceIdx_ = adjacentFaceKV.first;

      int32_t alphaMax = 3;
      for(int32_t alpha = 0; alpha <= alphaMax; alpha++)
      {
        float alphaRatio = static_cast<float>(alpha) / alphaMax;
        int32_t betaMax = std::round(alphaMax * (1.0 - alphaRatio));
        for(int32_t beta = 0; beta <= betaMax; beta++)
        {
          float betaRatio = static_cast<float>(beta) / betaMax * (1.0 - alphaRatio);
          ikTask.vertexWeights_.index_put_({0}, alphaRatio);
          ikTask.vertexWeights_.index_put_({1}, betaRatio);
          ikTask.vertexWeights_.index_put_({2}, 1.0 - (alphaRatio + betaRatio));

          torch::Tensor actualPos = ikTask.calcActualPos().to(torch::kCPU);
          torch::Tensor actualNormal = ikTask.calcActualNormal().to(torch::kCPU);

          geometry_msgs::Point startPointMsg;
          startPointMsg.x = actualPos.index({0}).item<float>();
          startPointMsg.y = actualPos.index({1}).item<float>();
          startPointMsg.z = actualPos.index({2}).item<float>();
          normalMarkerMsg.points.push_back(startPointMsg);

          geometry_msgs::Point endPointMsg = startPointMsg;
          endPointMsg.x += 0.02 * actualNormal.index({0}).item<float>();
          endPointMsg.y += 0.02 * actualNormal.index({1}).item<float>();
          endPointMsg.z += 0.02 * actualNormal.index({2}).item<float>();
          normalMarkerMsg.points.push_back(endPointMsg);
        }
      }
    }
    markerArrMsg.markers.push_back(normalMarkerMsg);

    g_clickedMarkerArrPub.publish(markerArrMsg);
  }
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
  bool enableQp = false;
  pnh.getParam("enable_qp", enableQp);
  bool solveMocapBody = false;
  pnh.getParam("solve_mocap_body", solveMocapBody);
  bool solveMocapMotion = false;
  pnh.getParam("solve_mocap_motion", solveMocapMotion);
  bool loadMotion = false;
  pnh.getParam("load_motion", loadMotion);
  bool enableVertexColor = false;
  pnh.getParam("enable_vertex_color", enableVertexColor);
  bool visualizeNormal = false;
  pnh.getParam("visualize_normal", visualizeNormal);
  bool visualizeSweepGrid = false;
  pnh.getParam("visualize_sweep_grid", visualizeSweepGrid);
  bool printDuration = true;
  bool solveMocap = solveMocapBody || solveMocapMotion;
  if(solveMocap)
  {
    enableVposer = !loadMotion;
    enableIk = !loadMotion;
    enableQp = true;
    printDuration = false;
  }

  ros::Publisher markerArrPub = nh.advertise<visualization_msgs::MarkerArray>("smplpp/marker_arr", 1);
  ros::Publisher targetPoseArrPub = nh.advertise<geometry_msgs::PoseArray>("smplpp/target_pose_arr", 1);
  ros::Publisher actualPoseArrPub = nh.advertise<geometry_msgs::PoseArray>("smplpp/actual_pose_arr", 1);
  ros::Publisher mocapMarkerArrPub;
  if(solveMocap)
  {
    mocapMarkerArrPub = nh.advertise<visualization_msgs::MarkerArray>("mocap/marker_arr", 1);
  }
  ros::Publisher gridCloudPub;
  if(visualizeSweepGrid)
  {
    gridCloudPub = nh.advertise<sensor_msgs::PointCloud>("smplpp/sweep_grid_cloud", 1);
  }
  g_clickedMarkerArrPub = nh.advertise<visualization_msgs::MarkerArray>("smplpp/marker_arr_clicked", 1);
  ros::Subscriber latentPoseParamSub;
  ros::Subscriber poseParamSub;
  if(!enableIk)
  {
    if(enableVposer)
    {
      latentPoseParamSub = nh.subscribe("smplpp/latent_pose_param", 1, latentPoseParamCallback);
    }
    else
    {
      poseParamSub = nh.subscribe("smplpp/pose_param", 1, poseParamCallback);
    }
  }
  ros::Subscriber shapeParamSub = nh.subscribe("smplpp/shape_param", 1, shapeParamCallback);
  ros::Subscriber ikTargetPoseSub = nh.subscribe("smplpp/ik_target_pose", 1, ikTargetPoseCallback);
  ros::Subscriber clickedPointSub = nh.subscribe("/clicked_point", 1, clickedPointCallback);

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

  // Setup variables
  g_theta = enableVposer ? torch::zeros({LATENT_POSE_DIM}) : torch::zeros({smplpp::JOINT_NUM + 1, 3});
  g_beta = torch::zeros({smplpp::SHAPE_BASIS_DIM});
  if(enableIk)
  {
    // Set initial root pose
    std::vector<double> initialPosVec = {0.0, 0.0, 0.0};
    pnh.getParam("initial_pos", initialPosVec);
    std::vector<double> initialRpyVec = {0.0, 0.0, 0.0};
    pnh.getParam("initial_rpy", initialRpyVec);
    Eigen::AngleAxisf initialAa(Eigen::AngleAxisf(initialRpyVec[2], Eigen::Vector3f::UnitZ())
                                * Eigen::AngleAxisf(initialRpyVec[1], Eigen::Vector3f::UnitY())
                                * Eigen::AngleAxisf(initialRpyVec[0], Eigen::Vector3f::UnitX()));
    torch::Tensor initialPosTensor =
        smplpp::toTorchTensor<float>(Eigen::Vector3d::Map(&initialPosVec[0]).cast<float>(), true);
    torch::Tensor initialRpyTensor = smplpp::toTorchTensor<float>(initialAa.angle() * initialAa.axis(), true);
    if(enableVposer)
    {
      g_theta.index_put_({at::indexing::Slice(0, 3)}, initialPosTensor);
      g_theta.index_put_({at::indexing::Slice(3, 6)}, initialRpyTensor);
    }
    else
    {
      g_theta.index_put_({0}, initialPosTensor);
      g_theta.index_put_({1}, initialRpyTensor);
    }
  }

  // Load SMPL model
  {
    auto startTime = std::chrono::system_clock::now();

    std::string smplPath;
    pnh.getParam("smpl_path", smplPath);
    ROS_INFO_STREAM("Load SMPL model from " << smplPath);

    g_smpl = std::make_shared<smplpp::SMPL>();
    g_smpl->setDevice(*device);
    g_smpl->setModelPath(smplPath);
    g_smpl->init();

    if(printDuration)
    {
      ROS_INFO_STREAM("Duration to load SMPL: " << std::chrono::duration_cast<std::chrono::duration<double>>(
                                                       std::chrono::system_clock::now() - startTime)
                                                       .count()
                                                << " [s]");
    }
  }

  // Load VPoser model
  smplpp::VPoserDecoder vposer;
  if(enableVposer)
  {
    auto startTime = std::chrono::system_clock::now();

    std::string vposerPath;
    pnh.getParam("vposer_path", vposerPath);
    ROS_INFO_STREAM("Load VPoser model from " << vposerPath);

    vposer->loadParamsFromJson(vposerPath);
    vposer->eval();
    vposer->to(*device);

    if(printDuration)
    {
      ROS_INFO_STREAM("Duration to load VPoser: " << std::chrono::duration_cast<std::chrono::duration<double>>(
                                                         std::chrono::system_clock::now() - startTime)
                                                         .count()
                                                  << " [s]");
    }
  }

  // Setup IK task list
  if(enableIk || loadMotion)
  {
    if(solveMocapBody)
    {
      // Ref. https://docs.optitrack.com/markersets/full-body/baseline-41
      g_ikTaskList.emplace("HeadTop", smplpp::IkTask(g_smpl, 7324));
      g_ikTaskList.emplace("HeadFront", smplpp::IkTask(g_smpl, 7194));
      g_ikTaskList.emplace("HeadSide", smplpp::IkTask(g_smpl, 13450));

      g_ikTaskList.emplace("Chest", smplpp::IkTask(g_smpl, 6842));
      g_ikTaskList.emplace("WaistLFront", smplpp::IkTask(g_smpl, 2162));
      g_ikTaskList.emplace("WaistRFront", smplpp::IkTask(g_smpl, 13026));
      g_ikTaskList.emplace("WaistLBack", smplpp::IkTask(g_smpl, 5117));
      g_ikTaskList.emplace("WaistRBack", smplpp::IkTask(g_smpl, 12007));
      g_ikTaskList.emplace("BackTop", smplpp::IkTask(g_smpl, 8914));
      g_ikTaskList.emplace("BackRight", smplpp::IkTask(g_smpl, 11433));
      g_ikTaskList.emplace("BackLeft", smplpp::IkTask(g_smpl, 4309));

      g_ikTaskList.emplace("LShoulderTop", smplpp::IkTask(g_smpl, 2261));
      g_ikTaskList.emplace("LShoulderBack", smplpp::IkTask(g_smpl, 4599));
      g_ikTaskList.emplace("LUArmHigh", smplpp::IkTask(g_smpl, 4249));
      g_ikTaskList.emplace("LElbowOut", smplpp::IkTask(g_smpl, 4913));
      g_ikTaskList.emplace("LWristIn", smplpp::IkTask(g_smpl, 4091));
      g_ikTaskList.emplace("LWristOut", smplpp::IkTask(g_smpl, 2567));
      g_ikTaskList.emplace("LHandOut", smplpp::IkTask(g_smpl, 2636));

      g_ikTaskList.emplace("RShoulderTop", smplpp::IkTask(g_smpl, 13583));
      g_ikTaskList.emplace("RShoulderBack", smplpp::IkTask(g_smpl, 11491));
      g_ikTaskList.emplace("RUArmHigh", smplpp::IkTask(g_smpl, 11137));
      g_ikTaskList.emplace("RElbowOut", smplpp::IkTask(g_smpl, 11802));
      g_ikTaskList.emplace("RWristIn", smplpp::IkTask(g_smpl, 9712));
      g_ikTaskList.emplace("RWristOut", smplpp::IkTask(g_smpl, 9590));
      g_ikTaskList.emplace("RHandOut", smplpp::IkTask(g_smpl, 9733));

      g_ikTaskList.emplace("LThigh", smplpp::IkTask(g_smpl, 1122));
      g_ikTaskList.emplace("LKneeOut", smplpp::IkTask(g_smpl, 1165));
      g_ikTaskList.emplace("LShin", smplpp::IkTask(g_smpl, 1247));
      g_ikTaskList.emplace("LAnkleOut", smplpp::IkTask(g_smpl, 5742));
      g_ikTaskList.emplace("LToeIn", smplpp::IkTask(g_smpl, 5758));
      g_ikTaskList.emplace("LToeOut", smplpp::IkTask(g_smpl, 6000));
      g_ikTaskList.emplace("LToeTip", smplpp::IkTask(g_smpl, 5591));
      g_ikTaskList.emplace("LHeel", smplpp::IkTask(g_smpl, 5815));

      g_ikTaskList.emplace("RThigh", smplpp::IkTask(g_smpl, 8523));
      g_ikTaskList.emplace("RKneeOut", smplpp::IkTask(g_smpl, 8053));
      g_ikTaskList.emplace("RShin", smplpp::IkTask(g_smpl, 12108));
      g_ikTaskList.emplace("RAnkleOut", smplpp::IkTask(g_smpl, 12630));
      g_ikTaskList.emplace("RToeIn", smplpp::IkTask(g_smpl, 12896));
      g_ikTaskList.emplace("RToeOut", smplpp::IkTask(g_smpl, 12889));
      g_ikTaskList.emplace("RToeTip", smplpp::IkTask(g_smpl, 12478));
      g_ikTaskList.emplace("RHeel", smplpp::IkTask(g_smpl, 12705));
    }
    else if(solveMocapMotion)
    {
      auto getDoubleValue = [](const XmlRpc::XmlRpcValue & param) -> double {
        return param.getType() == XmlRpc::XmlRpcValue::TypeInt ? static_cast<double>(static_cast<int>(param))
                                                               : static_cast<double>(param);
      };

      std::vector<double> betaVec;
      pnh.getParam("beta", betaVec);
      if(betaVec.size() != smplpp::SHAPE_BASIS_DIM)
      {
        throw smplpp::smpl_error("node", "Size of beta must be " + std::to_string(smplpp::SHAPE_BASIS_DIM) + " but "
                                             + std::to_string(betaVec.size()));
      }
      g_beta = torch::from_blob(std::vector<float>(betaVec.begin(), betaVec.end()).data(),
                                {static_cast<int64_t>(betaVec.size())})
                   .clone();

      XmlRpc::XmlRpcValue ikTaskListParam;
      pnh.getParam("ikTaskList", ikTaskListParam);
      for(int32_t paramIdx = 0; paramIdx < ikTaskListParam.size(); paramIdx++)
      {
        const XmlRpc::XmlRpcValue & ikTaskParam = ikTaskListParam[paramIdx];
        std::string name = static_cast<std::string>(ikTaskParam["name"]);
        int64_t faceIdx = static_cast<int>(ikTaskParam["faceIdx"]);
        const XmlRpc::XmlRpcValue & vertexWeightsParam = ikTaskParam["vertexWeights"];
        Eigen::Vector3d vertexWeightsVec;
        vertexWeightsVec << getDoubleValue(vertexWeightsParam[0]), getDoubleValue(vertexWeightsParam[1]),
            getDoubleValue(vertexWeightsParam[2]);
        smplpp::IkTask ikTask(g_smpl, faceIdx);
        ikTask.vertexWeights_ = smplpp::toTorchTensor<float>(vertexWeightsVec.cast<float>(), true);
        g_ikTaskList.emplace(name, ikTask);
      }
    }
    else
    {
      g_ikTaskList.emplace(
          "LeftHand", smplpp::IkTask(g_smpl, 2581, smplpp::toTorchTensor<float>(Eigen::Vector3f(0.5, 0.5, 1.0), true),
                                     smplpp::toTorchTensor<float>(Eigen::Vector3f::UnitZ(), true)));
      g_ikTaskList.emplace(
          "RightHand", smplpp::IkTask(g_smpl, 9469, smplpp::toTorchTensor<float>(Eigen::Vector3f(0.5, -0.5, 1.0), true),
                                      smplpp::toTorchTensor<float>(Eigen::Vector3f::UnitZ(), true)));
      g_ikTaskList.emplace(
          "LeftFoot", smplpp::IkTask(g_smpl, 5925, smplpp::toTorchTensor<float>(Eigen::Vector3f(0.0, 0.2, 0.0), true),
                                     smplpp::toTorchTensor<float>(Eigen::Vector3f::UnitZ(), true)));
      g_ikTaskList.emplace("RightFoot",
                           smplpp::IkTask(g_smpl, 12812,
                                          smplpp::toTorchTensor<float>(Eigen::Vector3f(0.0, -0.2, 0.0), true),
                                          smplpp::toTorchTensor<float>(Eigen::Vector3f::UnitZ(), true)));
    }

    if(solveMocap)
    {
      for(auto & ikTaskKV : g_ikTaskList)
      {
        auto & ikTask = ikTaskKV.second;
        ikTask.normalTaskWeight_ = 0.0;
        // Add offset of mocap marker thickness
        ikTask.normalOffset_ = 0.015; // [mm]
      }
    }

    for(auto & ikTaskKV : g_ikTaskList)
    {
      // ikTaskKV.second.normalTaskWeight_ = 0.0; // \todo Temporary
      ikTaskKV.second.phiLimit_ = 0.0; // \todo Temporary
    }
  }

  // Load C3D file
  std::shared_ptr<ezc3d::c3d> c3d;
  std::unordered_map<std::string, int32_t> ikMocapMarkerIdxMap;
  if(solveMocap)
  {
    std::string mocapPath;
    pnh.getParam("mocap_path", mocapPath);
    ROS_INFO_STREAM("Load mocap from " << mocapPath);

    c3d = std::make_shared<ezc3d::c3d>(mocapPath);

    const std::vector<std::string> & pointLabels =
        c3d->parameters().group("POINT").parameter("LABELS").valuesAsString();
    for(const auto & ikTaskKV : g_ikTaskList)
    {
      const auto & ikTaskName = ikTaskKV.first;
      auto checkEndSubstr = [&ikTaskName](const std::string & str) -> bool {
        return str.length() >= ikTaskName.length()
               && (str.compare(str.length() - ikTaskName.length(), ikTaskName.length(), ikTaskName) == 0);
      };
      auto findIter = std::find_if(pointLabels.cbegin(), pointLabels.cend(), checkEndSubstr);
      int32_t mocapMarkerIdx = std::distance(pointLabels.cbegin(), findIter);
      ikMocapMarkerIdxMap.emplace(ikTaskName, mocapMarkerIdx);
    }
  }

  int32_t mocapFrameIdx = 0;
  int32_t mocapFrameInterval = 1;
  if(solveMocapBody)
  {
    pnh.getParam("mocap_frame_idx", mocapFrameIdx);
  }
  else if(solveMocapMotion)
  {
    pnh.getParam("mocap_frame_interval", mocapFrameInterval);
  }

  smplpp::Motion motionMsg;
  if(solveMocapMotion)
  {
    if(loadMotion)
    {
      // Load message from rosbag
      std::string rosbagPath;
      pnh.getParam("rosbag_path", rosbagPath);
      ROS_INFO_STREAM("Load rosbag of mocap motion from " << rosbagPath);
      rosbag::Bag bag(rosbagPath, rosbag::bagmode::Read);
      for(const auto & msg : rosbag::View(bag))
      {
        if(msg.isType<smplpp::Motion>())
        {
          motionMsg = *(msg.instantiate<smplpp::Motion>());
          break;
        }
      }
    }
    else
    {
      motionMsg.frame_rate = c3d->header().frameRate() / static_cast<double>(mocapFrameInterval);
    }
  }

  double rateFreq = 30.0;
  pnh.getParam("rate", rateFreq);
  if(solveMocapMotion)
  {
    rateFreq = motionMsg.frame_rate;
    if(loadMotion)
    {
      rateFreq /= static_cast<double>(mocapFrameInterval);
    }
  }
  ros::Rate rate(rateFreq);

  for(int64_t ikIter = 0; ros::ok(); ikIter++)
  {
    // Update SMPL model and solve IK
    {
      std::vector<std::pair<std::string, double>> durationList;
      auto startTime = std::chrono::system_clock::now();

      bool optimizeBeta = false;
      if(solveMocapBody)
      {
        optimizeBeta = (ikIter >= 25);
      }

      // Update IK target from mocap
      if(loadMotion)
      {
        const smplpp::Instant & instantMsg = motionMsg.data_list[ikIter];
        mocapFrameIdx = instantMsg.frame_idx;
      }
      int32_t validMocapMarkerNum = 0;
      if(solveMocap)
      {
        const auto & points = c3d->data().frame(mocapFrameIdx).points();
        for(auto & ikTaskKV : g_ikTaskList)
        {
          auto & ikTask = ikTaskKV.second;
          int32_t mocapMarkerIdx = ikMocapMarkerIdxMap.at(ikTaskKV.first);

          const auto & point = points.point(mocapMarkerIdx);
          if(point.isEmpty())
          {
            if(solveMocapBody)
            {
              throw smplpp::smpl_error("node", "All mocap markers must be found to solve mocap body: " + ikTaskKV.first
                                                   + " not found.");
            }
            ikTask.posTaskWeight_ = 0.0;
            ikTask.targetPos_.zero_();
          }
          else
          {
            validMocapMarkerNum++;
            ikTask.posTaskWeight_ = 1.0;
            ikTask.targetPos_.index_put_({0}, point.x());
            ikTask.targetPos_.index_put_({1}, point.y());
            ikTask.targetPos_.index_put_({2}, point.z());
          }

          if(solveMocapBody)
          {
            ikTask.phiLimit_ = ikIter < 25 ? 0.0 : 0.04;
          }
          else if(solveMocapMotion)
          {
            ikTask.phiLimit_ = 0.0;
          }
        }
      }

      // Setup gradient
      if(enableIk)
      {
        g_theta.set_requires_grad(true);
        auto & g_theta_grad = g_theta.mutable_grad();
        if(g_theta_grad.defined())
        {
          g_theta_grad = g_theta_grad.detach();
          g_theta_grad.zero_();
        }

        for(auto & ikTaskKV : g_ikTaskList)
        {
          auto & ikTask = ikTaskKV.second;
          if(ikTask.phiLimit_ > 0.0)
          {
            ikTask.phi_.set_requires_grad(true);
            auto & phi_grad = ikTask.phi_.mutable_grad();
            if(phi_grad.defined())
            {
              phi_grad = phi_grad.detach();
              phi_grad.zero_();
            }
          }
          else
          {
            ikTask.phi_.set_requires_grad(false);
          }
        }

        if(optimizeBeta)
        {
          g_beta.set_requires_grad(true);
          auto & g_beta_grad = g_beta.mutable_grad();
          if(g_beta_grad.defined())
          {
            g_beta_grad = g_beta_grad.detach();
            g_beta_grad.zero_();
          }
        }
        else
        {
          g_beta.set_requires_grad(false);
        }
      }

      // Forward SMPL model
      {
        auto startTimeForward = std::chrono::system_clock::now();
        torch::Tensor theta;
        if(loadMotion)
        {
          const smplpp::Instant & instantMsg = motionMsg.data_list[ikIter];
          theta = smplpp::toTorchTensor<float>(
                      Eigen::VectorXd::Map(&instantMsg.theta[0], instantMsg.theta.size()).cast<float>(), true)
                      .view({smplpp::JOINT_NUM + 1, 3});
        }
        else if(enableVposer)
        {
          theta = torch::empty({smplpp::JOINT_NUM + 1, 3});
          theta.index_put_({0}, g_theta.index({at::indexing::Slice(0, 3)}));
          theta.index_put_({1}, g_theta.index({at::indexing::Slice(3, 6)}));
          torch::Tensor vposerIn =
              g_theta.index({at::indexing::Slice(6, smplpp::LATENT_DIM + 6)}).to(g_smpl->getDevice());
          torch::Tensor vposerOut = vposer->forward(vposerIn.view({1, -1})).index({0});
          theta.index_put_({at::indexing::Slice(2, 2 + 21)}, vposerOut);
          theta.index_put_({23}, g_theta.index({at::indexing::Slice(smplpp::LATENT_DIM + 6, smplpp::LATENT_DIM + 9)}));
          theta.index_put_({24}, g_theta.index({at::indexing::Slice(smplpp::LATENT_DIM + 9, smplpp::LATENT_DIM + 12)}));
        }
        else
        {
          theta = g_theta;
        }
        g_smpl->launch(g_beta.view({1, -1}), theta.view({1, theta.size(0), theta.size(1)}));
        durationList.emplace_back("forward SMPL", std::chrono::duration_cast<std::chrono::duration<double>>(
                                                      std::chrono::system_clock::now() - startTimeForward)
                                                          .count()
                                                      * 1e3);
      }

      // Solve IK
      if(enableIk && !(solveMocapMotion && validMocapMarkerNum < g_ikTaskList.size() / 2))
      {
        int32_t thetaDim = enableVposer ? LATENT_POSE_DIM : 3 * (smplpp::JOINT_NUM + 1);
        int32_t phiDim = 2 * g_ikTaskList.size();
        int32_t betaDim = optimizeBeta ? smplpp::SHAPE_BASIS_DIM : 0;
        Eigen::VectorXd e(4 * g_ikTaskList.size());
        Eigen::MatrixXd J(4 * g_ikTaskList.size(), thetaDim + phiDim + betaDim);
        J.middleCols(thetaDim, phiDim).setZero();
        int64_t rowIdx = 0;

        // Set end-effector position task
        auto startTimeSetupIk = std::chrono::system_clock::now();
        int32_t ikTaskIdx = 0;
        for(auto & ikTaskKV : g_ikTaskList)
        {
          auto & ikTask = ikTaskKV.second;

          // Update tangents and vertex weights
          ikTask.calcTangents();
          ikTask.calcVertexWeights(ikTask.calcActualPos().to(torch::kCPU).clone().detach());

          // Set task value
          torch::Tensor posError = ikTask.posTaskWeight_ * (ikTask.calcActualPos() - ikTask.targetPos_).to(torch::kCPU);
          e.segment<3>(rowIdx) = smplpp::toEigenMatrix(posError).cast<double>();

          torch::Tensor normalError;
          if(ikTask.normalTaskWeight_ > 0.0)
          {
            normalError = ikTask.normalTaskWeight_
                          * (at::dot(ikTask.calcActualNormal(), ikTask.targetNormal_).to(torch::kCPU) + 1.0);
            e.segment<1>(rowIdx + 3) = smplpp::toEigenMatrix(normalError.view({1})).cast<double>();
          }
          else
          {
            e.segment<1>(rowIdx + 3).setZero();
          }

          // Set task Jacobian
          for(int32_t i = 0; i < 3; i++)
          {
            // Calculate backward of each element
            torch::Tensor select = torch::zeros({1, 3});
            select.index_put_({0, i}, 1);
            posError.backward(select, true);

            J.row(rowIdx + i).head(thetaDim) =
                smplpp::toEigenMatrix(g_theta.grad().view({thetaDim}).to(torch::kCPU)).transpose().cast<double>();
            g_theta.mutable_grad().zero_();

            if(ikTask.phiLimit_ > 0.0)
            {
              J.row(rowIdx + i).segment<2>(thetaDim + 2 * ikTaskIdx) =
                  smplpp::toEigenMatrix(ikTask.phi_.grad().to(torch::kCPU)).transpose().cast<double>();
              ikTask.phi_.mutable_grad().zero_();
            }

            if(optimizeBeta)
            {
              J.row(rowIdx + i).tail(betaDim) =
                  smplpp::toEigenMatrix(g_beta.grad().view({betaDim}).to(torch::kCPU)).transpose().cast<double>();
              g_beta.mutable_grad().zero_();
            }
          }
          if(ikTask.normalTaskWeight_ > 0.0)
          {
            normalError.backward({}, true);

            J.row(rowIdx + 3).head(thetaDim) =
                smplpp::toEigenMatrix(g_theta.grad().view({thetaDim}).to(torch::kCPU)).transpose().cast<double>();
            g_theta.mutable_grad().zero_();

            if(ikTask.phiLimit_ > 0.0)
            {
              J.row(rowIdx + 3).segment<2>(thetaDim + 2 * ikTaskIdx) =
                  smplpp::toEigenMatrix(ikTask.phi_.grad().to(torch::kCPU)).transpose().cast<double>();
              ikTask.phi_.mutable_grad().zero_();
            }

            if(optimizeBeta)
            {
              J.row(rowIdx + 3).tail(betaDim) =
                  smplpp::toEigenMatrix(g_beta.grad().view({betaDim}).to(torch::kCPU)).transpose().cast<double>();
              g_beta.mutable_grad().zero_();
            }
          }
          else
          {
            J.row(rowIdx + 3).setZero();
          }

          rowIdx += 4;
          ikTaskIdx++;
        }
        durationList.emplace_back("calculate IK matrices", std::chrono::duration_cast<std::chrono::duration<double>>(
                                                               std::chrono::system_clock::now() - startTimeSetupIk)
                                                                   .count()
                                                               * 1e3);

        // Add regularization term
        Eigen::MatrixXd linearEqA = J.transpose() * J;
        Eigen::VectorXd linearEqB = J.transpose() * e;
        {
          constexpr double deltaThetaRegWeight = 1e-3;
          constexpr double deltaPhiRegWeight = 1e-1;
          constexpr double deltaBetaRegWeight = 1e-3;
          linearEqA.diagonal().head(thetaDim).array() += deltaThetaRegWeight;
          linearEqA.diagonal().segment(thetaDim, phiDim).array() += deltaPhiRegWeight;
          linearEqA.diagonal().tail(betaDim).array() += deltaBetaRegWeight;
          linearEqA.diagonal().array() += e.squaredNorm();
        }
        if(enableVposer)
        {
          constexpr double nominalPoseRegWeight = 1e-5;
          Eigen::VectorXd nominalPoseRegWeightVec = Eigen::VectorXd::Constant(thetaDim, nominalPoseRegWeight);
          nominalPoseRegWeightVec.head<6>().setZero();
          nominalPoseRegWeightVec.tail<6>().setConstant(1e3);
          linearEqA.diagonal().head(thetaDim) += nominalPoseRegWeightVec;
          linearEqB.head(thetaDim) +=
              nominalPoseRegWeightVec.cwiseProduct(smplpp::toEigenMatrix(g_theta.view({thetaDim})).cast<double>());
        }

        // Solve linear equation for IK
        auto startTimeSolveIk = std::chrono::system_clock::now();
        Eigen::VectorXd deltaConfig;
        if(enableQp)
        {
          auto qpSolver = QpSolverCollection::allocateQpSolver(QpSolverCollection::QpSolverType::QLD);
          QpSolverCollection::QpCoeff qpCoeff;
          qpCoeff.setup(thetaDim + phiDim + betaDim, 0, 0);
          qpCoeff.obj_mat_ = linearEqA;
          qpCoeff.obj_vec_ = linearEqB;
          ikTaskIdx = 0;
          for(auto & ikTaskKV : g_ikTaskList)
          {
            qpCoeff.x_min_.segment<2>(thetaDim + 2 * ikTaskIdx).setConstant(-1 * ikTaskKV.second.phiLimit_);
            qpCoeff.x_max_.segment<2>(thetaDim + 2 * ikTaskIdx).setConstant(ikTaskKV.second.phiLimit_);
            ikTaskIdx++;
          }
          if(optimizeBeta)
          {
            constexpr double deltaBetaLimit = 0.5;
            qpCoeff.x_min_.tail(betaDim).setConstant(-1 * deltaBetaLimit);
            qpCoeff.x_max_.tail(betaDim).setConstant(deltaBetaLimit);
          }
          deltaConfig = qpSolver->solve(qpCoeff);
        }
        else
        {
          Eigen::LLT<Eigen::MatrixXd> linearEqLlt(linearEqA);
          if(linearEqLlt.info() == Eigen::NumericalIssue)
          {
            throw smplpp::smpl_error("node", "LLT has numerical issue!");
          }
          deltaConfig = -1 * linearEqLlt.solve(linearEqB);
        }
        durationList.emplace_back("solve IK", std::chrono::duration_cast<std::chrono::duration<double>>(
                                                  std::chrono::system_clock::now() - startTimeSolveIk)
                                                      .count()
                                                  * 1e3);

        // Update config
        g_theta.set_requires_grad(false);
        g_theta += smplpp::toTorchTensor<float>(deltaConfig.head(thetaDim).cast<float>(), true).view_as(g_theta);

        Eigen::MatrixXf actualPosList(g_ikTaskList.size(), 3);
        ikTaskIdx = 0;
        for(auto & ikTaskKV : g_ikTaskList)
        {
          auto & ikTask = ikTaskKV.second;

          ikTask.phi_.set_requires_grad(false);
          torch::Tensor phi =
              smplpp::toTorchTensor<float>(deltaConfig.segment<2>(thetaDim + 2 * ikTaskIdx).cast<float>(), true);
          torch::Tensor actualPos = ikTask.calcActualPos().to(torch::kCPU) + torch::matmul(ikTask.tangents_, phi);
          actualPosList.row(ikTaskIdx) = smplpp::toEigenMatrix(actualPos).transpose();

          ikTaskIdx++;
        }

        if(optimizeBeta)
        {
          g_beta.set_requires_grad(false);
          g_beta += smplpp::toTorchTensor<float>(deltaConfig.tail(betaDim).cast<float>(), true).view_as(g_beta);
        }

        // Project point onto mesh
        Eigen::VectorXi closestFaceIndices;
        Eigen::MatrixX3f closestPoints;
        {
          auto startTimeProjectPoint = std::chrono::system_clock::now();

          torch::Tensor vertexTensor = g_smpl->getVertex().index({0}).to(torch::kCPU);
          Eigen::MatrixXf vertexMat = smplpp::toEigenMatrix(vertexTensor);
          torch::Tensor faceIdxTensor = g_smpl->getFaceIndex().to(torch::kCPU) - 1;
          Eigen::MatrixXi faceIdxMat = smplpp::toEigenMatrix<int>(faceIdxTensor);

          Eigen::VectorXf squaredDists;
          igl::point_mesh_squared_distance(actualPosList, vertexMat, faceIdxMat, squaredDists, closestFaceIndices,
                                           closestPoints);

          durationList.emplace_back("project point", std::chrono::duration_cast<std::chrono::duration<double>>(
                                                         std::chrono::system_clock::now() - startTimeProjectPoint)
                                                             .count()
                                                         * 1e3);
        }

        // Update face and vertex weights
        ikTaskIdx = 0;
        for(auto & ikTaskKV : g_ikTaskList)
        {
          auto & ikTask = ikTaskKV.second;

          ikTask.faceIdx_ = closestFaceIndices(ikTaskIdx);
          ikTask.calcVertexWeights(smplpp::toTorchTensor<float>(closestPoints.row(ikTaskIdx), true));

          ikTaskIdx++;
        }
      }

      if(printDuration)
      {
        ROS_INFO_STREAM_THROTTLE(10.0, "Duration to update SMPL: "
                                           << std::chrono::duration_cast<std::chrono::duration<double>>(
                                                  std::chrono::system_clock::now() - startTime)
                                                      .count()
                                                  * 1e3
                                           << " [ms]");
        std::string durationListStr;
        for(const auto & durationKV : durationList)
        {
          durationListStr +=
              "  - Duration to " + durationKV.first + ": " + std::to_string(durationKV.second) + " [ms]\n";
        }
        durationListStr.pop_back();
        ROS_INFO_STREAM_THROTTLE(10.0, durationListStr);
      }
    }

    // Calculate sweep grid
    if(visualizeSweepGrid)
    {
      auto startTime = std::chrono::system_clock::now();

      torch::Tensor vertexTensor = g_smpl->getVertex().index({0}).to(torch::kCPU);
      Eigen::MatrixXf vertexMat = smplpp::toEigenMatrix(vertexTensor);
      torch::Tensor faceIdxTensor = g_smpl->getFaceIndex().to(torch::kCPU) - 1;
      Eigen::MatrixXi faceIdxMat = smplpp::toEigenMatrix<int>(faceIdxTensor);

      Eigen::Vector3i gridIdxMin = smplpp::getGridIdxFloor<float>(vertexMat.colwise().minCoeff());
      Eigen::Vector3i gridIdxMax = smplpp::getGridIdxCeil<float>(vertexMat.colwise().maxCoeff());
      Eigen::Vector3i gridNum = gridIdxMax - gridIdxMin + Eigen::Vector3i::Ones();
      Eigen::MatrixXi gridIdxMat(gridNum[0] * gridNum[1] * gridNum[2], 3);
      int gridTotalIdx = 0;
      Eigen::Vector3i gridIdx;
      for(gridIdx[0] = gridIdxMin[0]; gridIdx[0] <= gridIdxMax[0]; gridIdx[0]++)
      {
        for(gridIdx[1] = gridIdxMin[1]; gridIdx[1] <= gridIdxMax[1]; gridIdx[1]++)
        {
          for(gridIdx[2] = gridIdxMin[2]; gridIdx[2] <= gridIdxMax[2]; gridIdx[2]++)
          {
            gridIdxMat.row(gridTotalIdx) = gridIdx.transpose();
            gridTotalIdx++;
          }
        }
      }
      Eigen::MatrixXf gridPosMat = smplpp::GRID_SCALE * gridIdxMat.cast<float>();

      Eigen::VectorXi windingNumbers;
      igl::winding_number(vertexMat, faceIdxMat, gridPosMat, windingNumbers);

      double timeNowSec = ros::Time::now().toSec();
      for(gridTotalIdx = 0; gridTotalIdx < windingNumbers.size(); gridTotalIdx++)
      {
        if(windingNumbers[gridTotalIdx] > 0.5)
        {
          g_sweepGridList[gridIdxMat.row(gridTotalIdx).transpose()] = timeNowSec;
        }
      }

      if(printDuration)
      {
        ROS_INFO_STREAM_THROTTLE(10.0, "Duration to calculate sweep grid: "
                                           << std::chrono::duration_cast<std::chrono::duration<double>>(
                                                  std::chrono::system_clock::now() - startTime)
                                                      .count()
                                                  * 1e3
                                           << " [ms]");
      }
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

      visualization_msgs::Marker normalMarkerMsg;
      if(visualizeNormal)
      {
        normalMarkerMsg.header = markerMsg.header;
        normalMarkerMsg.ns = "SMPL model normals";
        normalMarkerMsg.id = 1;
        normalMarkerMsg.type = visualization_msgs::Marker::LINE_LIST;
        normalMarkerMsg.action = visualization_msgs::Marker::ADD;
        normalMarkerMsg.pose.orientation.w = 1.0;
        normalMarkerMsg.scale.x = 0.002;
        normalMarkerMsg.scale.y = 0.002;
        normalMarkerMsg.scale.z = 0.002;
        normalMarkerMsg.color.r = 0.1;
        normalMarkerMsg.color.g = 0.1;
        normalMarkerMsg.color.b = 0.8;
        normalMarkerMsg.color.a = 1.0;
      }

      torch::Tensor vertexTensor = g_smpl->getVertex().index({0}).to(torch::kCPU);
      Eigen::MatrixX3d vertexMat = smplpp::toEigenMatrix(vertexTensor).cast<double>();
      torch::Tensor faceIdxTensor = g_smpl->getFaceIndex().to(torch::kCPU) - 1;
      Eigen::MatrixX3i faceIdxMat = smplpp::toEigenMatrix<int>(faceIdxTensor);

      double zMin = 0.0;
      double zMax = 0.0;
      if(enableVertexColor)
      {
        zMin = vertexMat.col(2).minCoeff();
        zMax = vertexMat.col(2).maxCoeff();
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

      for(int64_t faceIdx = 0; faceIdx < smplpp::FACE_INDEX_NUM; faceIdx++)
      {
        for(int32_t i = 0; i < 3; i++)
        {
          int64_t vertexIdx = faceIdxMat(faceIdx, i);
          geometry_msgs::Point pointMsg;
          pointMsg.x = vertexMat(vertexIdx, 0);
          pointMsg.y = vertexMat(vertexIdx, 1);
          pointMsg.z = vertexMat(vertexIdx, 2);
          markerMsg.points.push_back(pointMsg);
          if(enableVertexColor)
          {
            markerMsg.colors.push_back(makeColorMsg(pointMsg.z));
          }
        }

        if(visualizeNormal && faceIdx % 2 == 0)
        {
          Eigen::Vector3d centroid = (vertexMat.row(faceIdxMat(faceIdx, 0)) + vertexMat.row(faceIdxMat(faceIdx, 1))
                                      + vertexMat.row(faceIdxMat(faceIdx, 2)))
                                         .transpose()
                                     / 3.0;
          Eigen::Vector3d normal =
              centroid
              + 0.05
                    * (vertexMat.row(faceIdxMat(faceIdx, 1)) - vertexMat.row(faceIdxMat(faceIdx, 0)))
                          .transpose()
                          .cross((vertexMat.row(faceIdxMat(faceIdx, 2)) - vertexMat.row(faceIdxMat(faceIdx, 0)))
                                     .transpose())
                          .normalized();
          geometry_msgs::Point centroidPointMsg;
          centroidPointMsg.x = centroid.x();
          centroidPointMsg.y = centroid.y();
          centroidPointMsg.z = centroid.z();
          geometry_msgs::Point normalPointMsg;
          normalPointMsg.x = normal.x();
          normalPointMsg.y = normal.y();
          normalPointMsg.z = normal.z();
          normalMarkerMsg.points.push_back(centroidPointMsg);
          normalMarkerMsg.points.push_back(normalPointMsg);
        }
      }

      markerArrMsg.markers.push_back(markerMsg);
      if(visualizeNormal)
      {
        markerArrMsg.markers.push_back(normalMarkerMsg);
      }
      markerArrPub.publish(markerArrMsg);

      if(printDuration)
      {
        ROS_INFO_STREAM_THROTTLE(10.0, "Duration to publish SMPL model: "
                                           << std::chrono::duration_cast<std::chrono::duration<double>>(
                                                  std::chrono::system_clock::now() - startTime)
                                                      .count()
                                                  * 1e3
                                           << " [ms]");
      }
    }

    // Publish IK pose
    if(enableIk)
    {
      geometry_msgs::PoseArray targetPoseArrMsg;
      geometry_msgs::PoseArray actualPoseArrMsg;

      auto timeNow = ros::Time::now();
      targetPoseArrMsg.header.stamp = timeNow;
      targetPoseArrMsg.header.frame_id = "world";
      actualPoseArrMsg.header.stamp = timeNow;
      actualPoseArrMsg.header.frame_id = "world";

      for(const auto & ikTaskKV : g_ikTaskList)
      {
        const auto & ikTask = ikTaskKV.second;

        geometry_msgs::Pose targetPoseMsg;
        geometry_msgs::Pose actualPoseMsg;

        Eigen::Quaterniond targetQuat(smplpp::calcRotMatFromNormal(
            smplpp::toEigenMatrix(ikTask.targetNormal_.to(torch::kCPU)).cast<double>().normalized()));
        targetPoseMsg.position.x = ikTask.targetPos_.index({0}).item<float>();
        targetPoseMsg.position.y = ikTask.targetPos_.index({1}).item<float>();
        targetPoseMsg.position.z = ikTask.targetPos_.index({2}).item<float>();
        targetPoseMsg.orientation.w = targetQuat.w();
        targetPoseMsg.orientation.x = targetQuat.x();
        targetPoseMsg.orientation.y = targetQuat.y();
        targetPoseMsg.orientation.z = targetQuat.z();
        targetPoseArrMsg.poses.push_back(targetPoseMsg);

        torch::Tensor actualPos = ikTask.calcActualPos().to(torch::kCPU);
        torch::Tensor actualNormal = ikTask.calcActualNormal().to(torch::kCPU);
        Eigen::Quaterniond actualQuat(
            smplpp::calcRotMatFromNormal(smplpp::toEigenMatrix(actualNormal).cast<double>().normalized()));
        actualPoseMsg.position.x = actualPos.index({0}).item<float>();
        actualPoseMsg.position.y = actualPos.index({1}).item<float>();
        actualPoseMsg.position.z = actualPos.index({2}).item<float>();
        actualPoseMsg.orientation.w = actualQuat.w();
        actualPoseMsg.orientation.x = actualQuat.x();
        actualPoseMsg.orientation.y = actualQuat.y();
        actualPoseMsg.orientation.z = actualQuat.z();
        actualPoseArrMsg.poses.push_back(actualPoseMsg);
      }

      targetPoseArrPub.publish(targetPoseArrMsg);
      actualPoseArrPub.publish(actualPoseArrMsg);
    }

    // Publish mocap marker
    if(solveMocap)
    {
      visualization_msgs::MarkerArray markerArrMsg;
      visualization_msgs::Marker markerMsg;
      markerMsg.header.stamp = ros::Time::now();
      markerMsg.header.frame_id = "world";
      markerMsg.ns = "Mocap points";
      markerMsg.id = 0;
      markerMsg.type = visualization_msgs::Marker::SPHERE_LIST;
      markerMsg.action = visualization_msgs::Marker::ADD;
      markerMsg.pose.orientation.w = 1.0;
      markerMsg.scale.x = 0.04;
      markerMsg.scale.y = 0.04;
      markerMsg.scale.z = 0.04;
      markerMsg.color.r = 0.8;
      markerMsg.color.g = 0.1;
      markerMsg.color.b = 0.8;
      markerMsg.color.a = 1.0;

      const auto & points = c3d->data().frame(mocapFrameIdx).points();
      constexpr size_t mocapMarkerNum = 41;
      for(size_t mocapMarkerIdx = 0; mocapMarkerIdx < mocapMarkerNum; mocapMarkerIdx++)
      {
        const auto & point = points.point(mocapMarkerIdx);
        if(point.isEmpty())
        {
          continue;
        }

        geometry_msgs::Point pointMsg;
        pointMsg.x = point.x();
        pointMsg.y = point.y();
        pointMsg.z = point.z();
        markerMsg.points.push_back(pointMsg);
      }
      if(markerMsg.points.size() == 0)
      {
        markerMsg.action = visualization_msgs::Marker::DELETE;
      }

      visualization_msgs::Marker ikMarkerMsg;
      ikMarkerMsg.header = markerMsg.header;
      ikMarkerMsg.ns = "IK points";
      ikMarkerMsg.id = 1;
      ikMarkerMsg.type = visualization_msgs::Marker::SPHERE_LIST;
      ikMarkerMsg.action = visualization_msgs::Marker::ADD;
      ikMarkerMsg.pose.orientation.w = 1.0;
      ikMarkerMsg.scale.x = 0.04;
      ikMarkerMsg.scale.y = 0.04;
      ikMarkerMsg.scale.z = 0.04;
      ikMarkerMsg.color.r = 0.0;
      ikMarkerMsg.color.g = 0.8;
      ikMarkerMsg.color.b = 0.8;
      ikMarkerMsg.color.a = 1.0;

      for(const auto & ikTaskKV : g_ikTaskList)
      {
        torch::Tensor actualPos = ikTaskKV.second.calcActualPos().to(torch::kCPU);

        geometry_msgs::Point pointMsg;
        pointMsg.x = actualPos.index({0}).item<float>();
        pointMsg.y = actualPos.index({1}).item<float>();
        pointMsg.z = actualPos.index({2}).item<float>();
        ikMarkerMsg.points.push_back(pointMsg);
      }
      if(ikMarkerMsg.points.size() == 0)
      {
        ikMarkerMsg.action = visualization_msgs::Marker::DELETE;
      }

      markerArrMsg.markers.push_back(markerMsg);
      markerArrMsg.markers.push_back(ikMarkerMsg);
      mocapMarkerArrPub.publish(markerArrMsg);
    }

    // Publish sweep grid
    if(visualizeSweepGrid)
    {
      sensor_msgs::PointCloud cloudMsg;
      cloudMsg.header.stamp = ros::Time::now();
      cloudMsg.header.frame_id = "world";

      for(const auto & sweepGridKV : g_sweepGridList)
      {
        Eigen::Vector3f gridPos = smplpp::getGridPos(sweepGridKV.first);
        geometry_msgs::Point32 pointMsg;
        pointMsg.x = gridPos.x();
        pointMsg.y = gridPos.y();
        pointMsg.z = gridPos.z();
        cloudMsg.points.push_back(pointMsg);
      }

      gridCloudPub.publish(cloudMsg);
    }

    // Check terminal condition
    if(solveMocapBody)
    {
      if(ikIter % 10 == 0)
      {
        ROS_INFO_STREAM("[Iter " << ikIter << "] Solving mocap body.");
      }
      if(ikIter == 50)
      {
        break;
      }
    }
    else if(solveMocapMotion && loadMotion)
    {
      ikIter += mocapFrameInterval - 1;
      if(ikIter >= motionMsg.data_list.size() - 1)
      {
        break;
      }
    }
    else if(solveMocapMotion && !loadMotion)
    {
      if(ikIter % 10 == 0)
      {
        ROS_INFO_STREAM("[Iter " << ikIter << ", Frame " << mocapFrameIdx << " / " << c3d->header().nbFrames()
                                 << "] Solving mocap motion.");
      }
      if(ikIter > 30)
      {
        // Store data to message
        {
          Eigen::MatrixXd theta;
          if(enableVposer)
          {
            theta.resize(smplpp::JOINT_NUM + 1, 3);
            theta.row(0) = smplpp::toEigenMatrix(g_theta.index({at::indexing::Slice(0, 3)})).transpose().cast<double>();
            theta.row(1) = smplpp::toEigenMatrix(g_theta.index({at::indexing::Slice(3, 6)})).transpose().cast<double>();
            torch::Tensor vposerIn =
                g_theta.index({at::indexing::Slice(6, smplpp::LATENT_DIM + 6)}).to(g_smpl->getDevice());
            torch::Tensor vposerOut = vposer->forward(vposerIn.view({1, -1})).index({0});
            theta.middleRows(2, 21) = smplpp::toEigenMatrix(vposerOut).cast<double>();
            theta.row(23) = smplpp::toEigenMatrix(
                                g_theta.index({at::indexing::Slice(smplpp::LATENT_DIM + 6, smplpp::LATENT_DIM + 9)}))
                                .transpose()
                                .cast<double>();
            theta.row(24) = smplpp::toEigenMatrix(
                                g_theta.index({at::indexing::Slice(smplpp::LATENT_DIM + 9, smplpp::LATENT_DIM + 12)}))
                                .transpose()
                                .cast<double>();
          }
          else
          {
            theta = smplpp::toEigenMatrix(g_theta).cast<double>();
          }

          smplpp::Instant instantMsg;
          instantMsg.frame_idx = mocapFrameIdx;
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> thetaRowMajor = theta;
          instantMsg.theta.resize(thetaRowMajor.size());
          Eigen::VectorXd::Map(&instantMsg.theta[0], thetaRowMajor.size()) =
              Eigen::Map<const Eigen::VectorXd>(thetaRowMajor.data(), thetaRowMajor.size());
          motionMsg.data_list.push_back(instantMsg);
        }

        mocapFrameIdx += mocapFrameInterval;
      }
      if(mocapFrameIdx >= c3d->header().nbFrames())
      {
        break;
      }
    }

    ros::spinOnce();
    rate.sleep();

    if(loadMotion && rate.cycleTime().toSec() > rate.expectedCycleTime().toSec())
    {
      ROS_WARN_STREAM_THROTTLE(10.0, "Playback is delayed to real time. expected duration: "
                                         << rate.expectedCycleTime().toSec()
                                         << " [s], actual duration: " << rate.cycleTime().toSec() << " [s]");
    }
  }

  // Save mocap results
  if(solveMocapBody)
  {
    std::string mocapBodyPath = "/tmp/MocapBody.yaml";
    ROS_INFO_STREAM("Save IK task list for mocap to " << mocapBodyPath);
    std::ofstream ofs(mocapBodyPath);
    const Eigen::IOFormat fmt(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]", "", "");
    ofs << "beta: " << smplpp::toEigenMatrix(g_beta).transpose().format(fmt) << std::endl;
    ofs << "ikTaskList:" << std::endl;
    for(const auto & ikTaskKV : g_ikTaskList)
    {
      const auto & ikTask = ikTaskKV.second;
      ofs << "  - name: " << ikTaskKV.first << std::endl;
      ofs << "    faceIdx: " << ikTask.faceIdx_ << std::endl;
      ofs << "    vertexWeights: " << smplpp::toEigenMatrix(ikTask.vertexWeights_).transpose().format(fmt) << std::endl;
    }
  }
  else if(solveMocapMotion && !loadMotion)
  {
    std::string rosbagPath = "/tmp/MocapMotion.bag";
    ROS_INFO_STREAM("Save rosbag of mocap motion to " << rosbagPath);
    rosbag::Bag bag(rosbagPath, rosbag::bagmode::Write);
    bag.write("smplpp/motion", ros::Time::now(), motionMsg);
  }

  return 0;
}
