/* Author: Masaki Murooka */

#ifndef VPOSER_H
#define VPOSER_H

#include <torch/torch.h>

namespace smplpp
{
/** \brief VPoser decoder model.

    Corresponds to the model in
    https://github.com/nghorbani/human_body_prior/blob/master/src/human_body_prior/models/vposer_model.py
*/
class VPoserDecoderImpl : public torch::nn::Module
{
public:
  /** \brief Model to transform two 3D vectors to rotation matrices.

      Corresponds to the model in
      https://github.com/nghorbani/human_body_prior/blob/master/src/human_body_prior/models/vposer_model.py
  */
  class ContinousRotReprDecoderImpl : public torch::nn::Module
  {
  public:
    /** \brief Constructor. */
    ContinousRotReprDecoderImpl();

    /** \brief Forward model.
        \param input tensor representing two 3D vectors
        \returns tensor representing rotation matrices
    */
    torch::Tensor forward(const torch::Tensor & input);
  };

  /** \brief Model pointer to transform two 3D vectors to rotation matrices.

      See "Module Ownership" section of https://pytorch.org/tutorials/advanced/cpp_frontend.html for details.
  */
  TORCH_MODULE(ContinousRotReprDecoder);

public:
  /** \brief Constructor. */
  VPoserDecoderImpl();

  /** \brief Forward model.
      \param latent tensor representing latent variables
      \returns tensor representing joint angles
  */
  torch::Tensor forward(const torch::Tensor & latent);

public:
  //! Dimension of latent variables
  const int64_t latentDim_ = 32;

  //! Dimension of hidden variables
  const int64_t hiddenDim_ = 512;

  //! Number of joints
  const int64_t jointNum_ = 21;

  //! Decoder net
  torch::nn::Sequential decoderNet_;
};

/** \brief VPoser decoder model pointer.

    See "Module Ownership" section of https://pytorch.org/tutorials/advanced/cpp_frontend.html for details.
*/
TORCH_MODULE(VPoserDecoder);
} // namespace smplpp

#endif // VPOSER_H
