/* Author: Masaki Murooka */

#ifndef TORCH_EIGEN_UTILS_H
#define TORCH_EIGEN_UTILS_H

#include <Eigen/Dense>

#include <torch/torch.h>

#include <smplpp/toolbox/Exception.h>

namespace smplpp
{
/** \brief Row major version of Eige::MatrixXf. */
using MatrixXfRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
/** \brief Row major version of Eige::MatrixXd. */
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/** \brief Convert to torch::Tensor.
    \param mat input (Eigen::MatrixXf)
    \param tensor1d whether to convert to 1D tensor

    Even if the matrix of colum major is passed as an argument, it is automatically converted to row major.
*/
inline torch::Tensor toTorchTensor(const MatrixXfRowMajor & mat, bool tensor1d = false)
{
  if(tensor1d)
  {
    return torch::from_blob(const_cast<float *>(mat.data()), {mat.size()}).clone();
  }
  else
  {
    return torch::from_blob(const_cast<float *>(mat.data()), {mat.rows(), mat.cols()}).clone();
  }
}

/** \brief Convert to Eigen::MatrixXf.
    \param tensor input (torch::Tensor)
*/
inline Eigen::MatrixXf toEigenMatrix(const torch::Tensor & tensor)
{
  float * tensorDataPtr = const_cast<float *>(tensor.data_ptr<float>());
  if(tensor.dim() == 1)
  {
    return Eigen::Map<Eigen::VectorXf>(tensorDataPtr, tensor.size(0));
  }
  else if(tensor.dim() == 2)
  {
    return Eigen::MatrixXf(Eigen::Map<MatrixXfRowMajor>(tensorDataPtr, tensor.size(0), tensor.size(1)));
  }
  else
  {
    throw smpl_error("TorchEigenUtils", "Invalid tensor dimension: " + std::to_string(tensor.dim()));
  }
}
} // namespace smplpp

#endif // TORCH_EIGEN_UTILS_H
