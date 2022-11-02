/* Author: Masaki Murooka */

#ifndef IK_TASK_H
#define IK_TASK_H

#include <torch/torch.h>

namespace smplpp
{
class SMPL;

/** \brief IK task. */
class IkTask
{
public:
  /** \brief Constructor.
      \param smpl SMPL instance
      \param faceIdx index of face to be focused on in IK
  */
  IkTask(const std::shared_ptr<smplpp::SMPL> & smpl, int64_t faceIdx);

  /** \brief Constructor.
      \param smpl SMPL instance
      \param faceIdx index of face to be focused on in IK
      \param targetPos target position
      \param targetNormal target unit normal vector
  */
  IkTask(const std::shared_ptr<smplpp::SMPL> & smpl,
         int64_t faceIdx,
         torch::Tensor targetPos,
         torch::Tensor targetNormal);

  /** \brief Calculate tangent vectors of the focused face.

      Tangent vectors are detached from the computation graph, and their gradients are not considered.
  */
  void calcTangents();

  /** \brief Calculate vertex weights.
      \param actualPos point position on the focused face

      Calculates vertex weights such that actualPos is represented by a weighted sum of vertices.
  */
  void calcVertexWeights(const torch::Tensor & actualPos);

  /** \brief Calculate the position. */
  torch::Tensor calcActualPos() const;

  /** \brief Calculate the unit normal vector. */
  torch::Tensor calcActualNormal() const;

public:
  //! SMPL instance
  std::shared_ptr<smplpp::SMPL> smpl_;

  //! Index of face to be focused on in IK
  int64_t faceIdx_;

  //! Weight of position task
  double posTaskWeight_ = 1.0;

  //! Weight of normal task
  double normalTaskWeight_ = 1.0;

  //! Limit of phi [m]
  double phiLimit_ = 0.04;

  //! Normal offset of position task [m]
  double normalOffset_ = 0.0;

  //! Target of position task
  torch::Tensor targetPos_;

  //! Target of normal task
  torch::Tensor targetNormal_;

  //! Vertex weights
  torch::Tensor vertexWeights_ = torch::empty({3}).fill_(1.0 / 3.0);

  //! Tangent vectors
  torch::Tensor tangents_ = torch::zeros({3, 2});

  //! Phi (i.e., displacement along tangent vectors) [m]
  torch::Tensor phi_ = torch::zeros({2});
};
} // namespace smplpp

#endif // IK_TASK_H
