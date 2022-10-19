#!/bin/sh

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_c3d_dir> <output_rosbag_dir>"
  exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2

FILENAME_LIST=$(basename --suffix=.c3d $(ls ${INPUT_DIR}/*.c3d))

echo "[solveMocapMotionLoop] Filename list:"
for FILENAME in ${FILENAME_LIST}; do
    echo "  - $FILENAME"
done

for FILENAME in ${FILENAME_LIST}; do
    C3D_PATH=${INPUT_DIR}/${FILENAME}.c3d
    ROSBAG_PATH=${OUTPUT_DIR}/${FILENAME}.bag
    echo "[solveMocapMotionLoop] Process ${FILENAME}"
    echo "  - Input: ${C3D_PATH}"
    echo "  - Output: ${ROSBAG_PATH}"
    time roslaunch smplpp smplpp.launch solve_mocap_motion:=true mocap_path:=${C3D_PATH}
    mv /tmp/MocapMotion.bag ${ROSBAG_PATH}
done
