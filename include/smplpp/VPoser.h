/* Author: Masaki Murooka */

#ifndef VPOSER_H
#define VPOSER_H

#include <torch/torch.h>

namespace smplpp
{
/** \brief VPoser decoder model.

    Corresponds to decoder model in
   https://github.com/nghorbani/human_body_prior/blob/master/src/human_body_prior/models/vposer_model.py
*/
class VPoserDecoderImpl : public torch::nn::Module
{
public:
  /** \brief Constructor. */
  VPoserDecoderImpl();

  /** \brief Forward model.
      \param latent latent variables
      \returns joint angles
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
