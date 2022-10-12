#!/usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import pickle as pkl
import json
import torch


def main(args):
    """Main function to convert the file of VPoser parameters from PyTorch format to NumPy and JSON formats.

    Arguments
    ----------
    - args: list of strings
        Command line arguments.

    Returns
    ----------

    """
    model_load_path = args[1]
    save_dir_path = args[2]

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    np_save_path = os.path.join(save_dir_path, "vposer_parameters.npz")
    json_save_path = os.path.join(save_dir_path, "vposer_parameters.json")

    print("Load VPoser parameters from: ", os.path.abspath(model_load_path))
    with open(model_load_path, "rb") as f:
        raw_model_data = torch.load(f)["state_dict"]

    model_data_np = {}
    model_data_json = {}
    key_list = ["decoder_net.0.weight",
                "decoder_net.0.bias",
                "decoder_net.3.weight",
                "decoder_net.3.bias",
                "decoder_net.5.weight",
                "decoder_net.5.bias"]
    for key in key_list:
        mat = np.array(raw_model_data["vp_model.{}".format(key)].cpu())
        model_data_np[key] = mat
        # Data must be converted to list before storing as json.
        model_data_json[key] = mat.tolist()

    print("Save VPoser parameters to: ", os.path.abspath(save_dir_path))
    np.savez(np_save_path, **model_data_np)
    with open(json_save_path, "wb+") as f:
        json.dump(model_data_json, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    if sys.version_info[0] != 2:
        raise EnvironmentError("Run this file with Python2!")
    if len(sys.argv) != 3:
        raise SystemError("Invalid number of arguments!\n"
                          "USAGE: python2 preprocess.py "
                          "<model_load_path> <save_dir_path>")

    main(sys.argv)
