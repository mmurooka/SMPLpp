/* Author: Masaki Murooka */

#include <smplpp/IkTask.h>
#include <smplpp/SMPL.h>
#include <smplpp/toolbox/GeometryUtils.h>

using namespace smplpp;

IkTask::IkTask(const std::shared_ptr<smplpp::SMPL> & smpl, int64_t faceIdx) : smpl_(smpl), faceIdx_(faceIdx)
{
  targetPos_ = torch::zeros({3});
  targetNormal_ = torch::empty({3});
  targetNormal_.index_put_({0}, 0.0);
  targetNormal_.index_put_({1}, 0.0);
  targetNormal_.index_put_({2}, 1.0);

  targetPos_ = targetPos_.to(smpl_->getDevice());
  targetNormal_ = targetNormal_.to(smpl_->getDevice());
  vertexWeights_ = vertexWeights_.to(smpl_->getDevice());
}

IkTask::IkTask(const std::shared_ptr<smplpp::SMPL> & smpl,
               int64_t faceIdx,
               torch::Tensor targetPos,
               torch::Tensor targetNormal)
: smpl_(smpl), faceIdx_(faceIdx), targetPos_(targetPos), targetNormal_(targetNormal)
{
  targetPos_ = targetPos_.to(smpl_->getDevice());
  targetNormal_ = targetNormal_.to(smpl_->getDevice());
  vertexWeights_ = vertexWeights_.to(smpl_->getDevice());
}

void IkTask::calcTangents()
{
  torch::Tensor faceVertexIdxs = smpl_->getFaceIndexRaw(faceIdx_).to(torch::kCPU) - 1;
  // Clone and detach vertex tensors to separate tangents from the computation graph
  torch::Tensor faceVertices = smpl_->getVertexRaw(faceVertexIdxs.to(torch::kInt64)).to(torch::kCPU).clone().detach();

  torch::Tensor tangent1 = faceVertices.index({1}) - faceVertices.index({0});
  torch::Tensor normal = at::cross(tangent1, faceVertices.index({2}) - faceVertices.index({0}));
  torch::Tensor tangent2 = at::cross(normal, tangent1);
  tangent1 = torch::nn::functional::normalize(tangent1, torch::nn::functional::NormalizeFuncOptions().dim(-1));
  tangent2 = torch::nn::functional::normalize(tangent2, torch::nn::functional::NormalizeFuncOptions().dim(-1));
  tangents_.index_put_({at::indexing::Slice(), 0}, tangent1);
  tangents_.index_put_({at::indexing::Slice(), 1}, tangent2);
  tangents_ = tangents_.clone().detach();
}

void IkTask::calcVertexWeights(const torch::Tensor & actualPos)
{
  torch::Tensor faceVertexIdxs = smpl_->getFaceIndexRaw(faceIdx_).to(torch::kCPU) - 1;
  // Clone and detach vertex tensors to separate tangents from the computation graph
  torch::Tensor faceVertices = smpl_->getVertexRaw(faceVertexIdxs.to(torch::kInt64)).to(torch::kCPU).clone().detach();

  torch::Tensor pos = actualPos + torch::matmul(tangents_, phi_);
  vertexWeights_ = smplpp::calcTriangleVertexWeights(pos, faceVertices).to(smpl_->getDevice());
}

torch::Tensor IkTask::calcActualPos() const
{
  torch::Tensor faceVertexIdxs = smpl_->getFaceIndexRaw(faceIdx_).to(torch::kCPU) - 1;
  torch::Tensor faceVertices = smpl_->getVertexRaw(faceVertexIdxs.to(torch::kInt64));

  torch::Tensor actualPos = torch::matmul(torch::transpose(faceVertices, 0, 1), vertexWeights_);

  if(normalOffset_ > 0.0)
  {
    actualPos += normalOffset_ * calcActualNormal();
  }

  return actualPos;
}

torch::Tensor IkTask::calcActualNormal() const
{
  torch::Tensor faceVertexIdxs = smpl_->getFaceIndexRaw(faceIdx_).to(torch::kCPU) - 1;

  torch::Tensor actualNormal = torch::zeros({3}, smpl_->getDevice());
  for(int32_t i = 0; i < 3; i++)
  {
    int32_t faceVertexIdx = faceVertexIdxs.index({i}).item<int32_t>();
    actualNormal += vertexWeights_.index({i}) * smpl_->calcVertexNormal(faceVertexIdx);
  }

  return torch::nn::functional::normalize(actualNormal, torch::nn::functional::NormalizeFuncOptions().dim(-1));
}
