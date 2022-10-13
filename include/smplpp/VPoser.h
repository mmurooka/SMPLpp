/* Author: Masaki Murooka */

#ifndef VPOSER_H
#define VPOSER_H

#include <torch/torch.h>

#include <smplpp/definition/def.h>

namespace smplpp
{
/** \brief Convert rotation matrices to axis-angle representations.
    \param rotMat tensor representing rotation matrices (N, 3, 3)
    \returns tensor representing axis-angle representations (N, 3)

    See
    https://github.com/jrl-umi3218/SpaceVecAlg/blob/676a64c47d650ba5de6a4b0ff4f2aaf7262ffafe/src/SpaceVecAlg/PTransform.h#L293-L342
    for the algorithm.
*/
torch::Tensor convertRotMatToAxisAngle(const torch::Tensor & rotMat);

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
        \param input tensor representing two 3D vectors (N, 6 * jointNum_)
        \returns tensor representing rotation matrices (N, 3, 3)
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
      \param latent tensor representing latent variables (B, LATENT_DIM)
      \returns tensor representing joint angles (B, jointNum_, 3)
  */
  torch::Tensor forward(const torch::Tensor & latent);

  /** \brief Load model parameters from JSON file.
      \param jsonPath Path of JSON file
  */
  void loadParamsFromJson(const std::string & jsonPath);

public:
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
