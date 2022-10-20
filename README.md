# [SMPLpp](https://github.com/mmurooka/SMPLpp)
Inverse kinematics with 3D human model SMPL

## Overview
This repository [mmurooka/SMPLpp](https://github.com/mmurooka/SMPLpp) is a fork of [YeeCY/SMPLpp](https://github.com/YeeCY/SMPLpp).
Most of the source code is identical to the original repository, and their licenses follow the original repository.
Newly added source code in this repository is available under [MIT license](https://github.com/mmurooka/SMPLpp/blob/master/LICENSE), as in the original repository.

This repository contains the following updates and extensions to the original repository:
- C++ implementation of inverse kinematics (MoSh, MoSh++)
- C++ implementation of body pose prior (VPoser)
- Minor bug fixes (https://github.com/YeeCY/SMPLpp/pull/13, https://github.com/YeeCY/SMPLpp/pull/14)
- ROS packaging

For more information on the technical methods, please see the following papers:
- 3D human model: M. Loper, et al. SMPL: A skinned multi-person linear model. ToG, 2015.
- Body pose prior: G. Pavlakos, et al. Expressive body capture: 3d hands, face, and body from a single image. CVPR, 2019.
- Inverse kinematics: M. Loper, et al. MoSh: Motion and shape capture from sparse markers. ToG, 2014.
- Inverse kinematics: M. Naureen, et al. AMASS: Archive of motion capture as surface shapes. ICCV, 2019.

This repository does not contain model parameter files.
They can be downloaded from the following project pages of the paper authors.
Please see the project page for the license.
- SMPL: https://smpl.is.tue.mpg.de/
- VPoser https://smpl-x.is.tue.mpg.de/

## Install

### Requirements
- Compiler supporting C++14
- Tested on `Ubuntu 18.04 / ROS Melodic`

### Dependencies
This package depends on
- [xtl](https://github.com/xtensor-stack/xtl)
- [xtensor](https://github.com/xtensor-stack/xtensor)
- [nlohmann-json](https://github.com/nlohmann/json)
- [ezc3d](https://github.com/pyomeca/ezc3d)
- [eigen-qld](https://github.com/jrl-umi3218/eigen-qld)
- [libtorch (PyTorch C++ Frontend)](https://pytorch.org/cppdocs/installing.html)
- [libigl](https://libigl.github.io/)
  - libigl is automatically downloaded during the build of this package, so there is no need to install it manually.
- [QpSolverCollection](https://github.com/isri-aist/QpSolverCollection)

### Installation procedure
It is assumed that ROS is installed.

1. Install dependent packages.

For [xtl](https://github.com/xtensor-stack/xtl), [xtensor](https://github.com/xtensor-stack/xtensor), [nlohmann-json](https://github.com/nlohmann/json), [ezc3d](https://github.com/pyomeca/ezc3d), and [eigen-qld](https://github.com/jrl-umi3218/eigen-qld), download the source code from GitHub and install it with cmake.

For libtorch, follow [the official instructions](https://pytorch.org/cppdocs/installing.html) to download and extract the zip file.

2. Setup catkin workspace.
```bash
$ mkdir -p ~/ros/ws_smplpp/src
$ cd ~/ros/ws_smplpp
$ wstool init src
$ wstool set -t src isri-aist/QpSolverCollection git@github.com:isri-aist/QpSolverCollection.git --git -y
$ wstool set -t src mmurooka/SMPLpp git@github.com:mmurooka/SMPLpp.git --git -y
$ wstool update -t src
```

3. Install dependent packages.
```bash
$ source /opt/ros/${ROS_DISTRO}/setup.bash
$ rosdep install -y -r --from-paths src --ignore-src
```

4. Build a package.
```bash
$ catkin build smplpp -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLIBTORCH_PATH=<absolute path to libtorch> -DENABLE_QLD=ON --catkin-make-args all tests
```
`<absolute path to libtorch>` is the path to the directory named libtorch that was extracted in step 1.

## Examples
