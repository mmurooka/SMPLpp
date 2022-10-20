# [SMPLpp](https://github.com/mmurooka/SMPLpp)
Inverse kinematics with 3D human model SMPL

## Overview
This repository [mmurooka/SMPLpp](https://github.com/mmurooka/SMPLpp) is a fork of [YeeCY/SMPLpp](https://github.com/YeeCY/SMPLpp).
This README, written by [mmurooka](https://github.com/mmurooka), describes [mmurooka/SMPLpp](https://github.com/mmurooka/SMPLpp).

Most of the source code is identical to the original repository, and their licenses follow the original repository.
Newly added source code in this repository is available under [MIT license](https://github.com/mmurooka/SMPLpp/blob/master/LICENSE), as in the original repository.

This repository contains the following updates and extensions to the original repository:
- C++ implementation of inverse kinematics (MoSh, MoSh++)
- C++ implementation of body pose prior (VPoser)
- Function for fitting SMPL models to motion capture data
- Minor bug fixes (https://github.com/YeeCY/SMPLpp/pull/13, https://github.com/YeeCY/SMPLpp/pull/14)
- ROS packaging

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

## Samples

### Prerequisites: download and preprocessing of model parameters

1. Download model parameter files.

Download the model parameter file from the `Download version 1.0.0 for Python 2.7 (female/male. 10 shape PCs)` link on the following project page of the paper authors. Please note the license on the project page.
- https://smpl.is.tue.mpg.de/

Download the model parameter file from the `Download VPoser v2.0 - not-evaluated (80.0 MB)` link on the following project page of the paper authors. Please note the license on the project page.
- https://smpl-x.is.tue.mpg.de/

2. Extract the downloaded zip files.

Extract the zip file `SMPL_python_v.1.0.0.zip`.
The extracted zip files contain the model parameter files: `basicmodel_m_lbs_10_207_0_v1.0.0.pkl` and `basicModel_f_lbs_10_207_0_v1.0.0.pkl`.

Extract the zip file `V02_05.zip`.
The extracted zip files contain the model parameter file: `V02_05_epoch=08_val_loss=0.03.ckpt`.

3. Preprocess SMPL model parameter files.
```bash
$ rosrun smplpp preprocess.py male <path to basicmodel_m_lbs_10_207_0_v1.0.0.pkl> `rospack find smplpp`/data
$ rosrun smplpp preprocess.py female <path to basicModel_f_lbs_10_207_0_v1.0.0.pkl> `rospack find smplpp`/data
```
Confirm that the files `smpl_male.{json|npz}` and `smpl_female.{json|npz}` have been generated in `rospack find smplpp`/data.

4. Preprocess VPoser model parameter file.
```bash
$ rosrun smplpp preprocess_vposer.py <path to V02_05_epoch=08_val_loss=0.03.ckpt> `rospack find smplpp`/data
```
Confirm that the files `vposer_parameters.{json|npz}` have been generated in `rospack find smplpp`/data.

### Forward kinematics
```bash
$ roslaunch smplpp smplpp.launch enable_vertex_color:=false enable_ik:=false enable_vposer:=false
```
You will see a window of Rviz and two windows of rqt.
You can change the SMPL body parameters (`beta`) and pose parameters (`theta`) from the sliders in the two rqt windows, and watch the SMPL model visualized in Rviz change.

If you set `enable_vposer:=true` instead of `enable_vposer:=false` in the `roslaunch` arguments, you can change the latent variables from the sliders instead of the pose parameters.

https://user-images.githubusercontent.com/6636600/196848840-437fe145-9998-4f1b-8c49-deb2677c69c0.mp4

### Inverse kinematics
```bash
$ roslaunch smplpp smplpp.launch enable_vertex_color:=false enable_ik:=true enable_vposer:=true
```
You will see a window of Rviz and a window of rqt.
By moving the interactive markers on Rviz, you can specify the position and normal direction of the hands and feet of the SMPL model.
You can change the SMPL body parameters (`beta`) from the sliders in the rqt window.

If you set `enable_vposer:=false` instead of `enable_vposer:=true` in the `roslaunch` arguments, the pose parameters can be optimized directly instead of the latent variables. In this case, the body pose prior is not taken into account and the generated pose will be unnatural.

https://user-images.githubusercontent.com/6636600/196848863-971793eb-a620-4266-837d-e90cc9360303.mp4

### Motion capture fittings

### Notes
#### Note on vertex-colored meshes in Rviz
If you set `enable_vertex_color:=true` instead of `enable_vertex_color:=false` in the `roslaunch` arguments, the color of each vertex is determined by its Z-position. However, Rviz on the default branch does not provide good visualization for vertex-colored meshes. Building Rviz from source code with [this patch](https://github.com/mmurooka/rviz/commit/ad4ac3206c2e72c073498b4568edeeff0c256f82) solves this problem.

#### Interactive retrieval of vertex and face indices
When you click the `Publish Point` button in Rviz and then click on the SMPL model, the indices of the vertex and face closest to the clicked point are printed in the terminal as follows. This is useful for determining the inverse-kinematics target on the SMPL model.
```bash
[ INFO][/smplpp]: Vertex idx closest to clicked point: 6875
[ INFO][/smplpp]: Face idx closest to clicked point: 13353
```

## Technical details
For more information on the technical details, please see the following papers:
- 3D human model: M. Loper, et al. SMPL: A skinned multi-person linear model. ToG, 2015.
- Body pose prior: G. Pavlakos, et al. Expressive body capture: 3d hands, face, and body from a single image. CVPR, 2019.
- Inverse kinematics: M. Loper, et al. MoSh: Motion and shape capture from sparse markers. ToG, 2014.
- Inverse kinematics: M. Naureen, et al. AMASS: Archive of motion capture as surface shapes. ICCV, 2019.
