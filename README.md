# [SMPLpp](https://github.com/mmurooka/SMPLpp)
Inverse kinematics with 3D human model SMPL

## Overview
This repository [mmurooka/SMPLpp](https://github.com/mmurooka/SMPLpp) is a fork of [YeeCY/SMPLpp](https://github.com/YeeCY/SMPLpp).
Most of the source code is identical to the original repository, and their licenses follow the original repository.
Newly added source code in this repository is available under [MIT license](https://github.com/mmurooka/SMPLpp/blob/master/LICENSE), as in the original repository.

This repository contains the following updates and extensions to the original repository:
- C++ implementation of inverse kinematics (MoSh, MoSh++)
- C++ implementation of body pose prior (VPoser)
- Minor bug fixes (https://github.com/YeeCY/SMPLpp/pull/13 https://github.com/YeeCY/SMPLpp/pull/14)
- ROS packaging

For more information on the technical methods, please see the following papers:
- 3D human model: M. Loper, et al. SMPL: A skinned multi-person linear model. ToG, 2015.
- Body pose prior: G. Pavlakos, et al. Expressive body capture: 3d hands, face, and body from a single image. CVPR, 2019.
- Inverse kinematics: M. Loper, et al. MoSh: Motion and shape capture from sparse markers. ToG, 2014.
- Inverse kinematics: M. Naureen, et al. AMASS: Archive of motion capture as surface shapes. ICCV, 2019.
