/* Author: Masaki Murooka */

#ifndef TORCH_EIGEN_UTILS_H
#define TORCH_EIGEN_UTILS_H

#include <Eigen/Dense>

#include <torch/torch.h>

#include <smplpp/toolbox/Exception.h>

namespace smplpp
{
/** \brief Convert to torch tensor.
    \param mat Eigen matrix
    \param tensor1d whether to convert to 1D tensor

    Even if the matrix of column major is passed as an argument, it is automatically converted to row major.
*/
template<typename Scalar = float>
inline torch::Tensor toTorchTensor(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & mat,
                                   bool tensor1d = false)
{
  if(tensor1d)
  {
    return torch::from_blob(const_cast<Scalar *>(mat.data()), {mat.size()}).clone();
  }
  else
  {
    return torch::from_blob(const_cast<Scalar *>(mat.data()), {mat.rows(), mat.cols()}).clone();
  }
}

/** \brief Convert to Eigen matrix.
    \param tensor torch tensor
*/
template<typename Scalar = float>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> toEigenMatrix(const torch::Tensor & tensor)
{
  Scalar * tensorDataPtr = const_cast<Scalar *>(tensor.data_ptr<Scalar>());
  if(tensor.dim() == 1)
  {
    return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(tensorDataPtr, tensor.size(0));
  }
  else if(tensor.dim() == 2)
  {
    return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            tensorDataPtr, tensor.size(0), tensor.size(1)));
  }
  else
  {
    throw smpl_error("TorchEigenUtils", "Invalid tensor dimension: " + std::to_string(tensor.dim()));
  }
}
} // namespace smplpp

#endif // TORCH_EIGEN_UTILS_H
