/* Author: Masaki Murooka */

#ifndef GEOMETRY_UTILS_H
#define GEOMETRY_UTILS_H

#include <Eigen/Dense>

#include <torch/torch.h>

#include <smplpp/toolbox/Exception.h>

namespace smplpp
{
/** \brief Calculate rotation matrix from unit normal vector (Z-axis).
    \param normal unit normal vector (Z-axis)
*/
Eigen::Matrix3d calcRotMatFromNormal(const Eigen::Vector3d & normal)
{
  Eigen::Vector3d tangent1;
  if(std::abs(normal.dot(Eigen::Vector3d::UnitX())) < 1.0 - 1e-3)
  {
    tangent1 = Eigen::Vector3d::UnitX().cross(normal).normalized();
  }
  else
  {
    tangent1 = Eigen::Vector3d::UnitY().cross(normal).normalized();
  }
  Eigen::Vector3d tangent2 = normal.cross(tangent1).normalized();
  Eigen::Matrix3d rotMat;
  rotMat << tangent1, tangent2, normal;
  return rotMat;
}

/** \brief Calculate weights of triangle vertices
    \param pos position of focused point, which should be inside or on the boundary of the triangle
    \param vertices triangle vertices

    Ref.
      - https://www.geisya.or.jp/~mwm48961/koukou/complex_line2.htm
      - https://homeskill.hatenadiary.org/entry/20110112/1294831298
*/
torch::Tensor calcTriangleVertexWeights(const torch::Tensor & pos, const torch::Tensor & vertices)
{
  torch::Tensor weights = torch::empty({3});

  weights.index_put_({0}, at::cross(vertices.index({1}) - pos, vertices.index({2}) - pos).norm());
  weights.index_put_({1}, at::cross(vertices.index({2}) - pos, vertices.index({0}) - pos).norm());
  weights.index_put_({2}, at::cross(vertices.index({0}) - pos, vertices.index({1}) - pos).norm());
  weights /= weights.sum();

  return weights;
}
} // namespace smplpp

#endif // GEOMETRY_UTILS_H
