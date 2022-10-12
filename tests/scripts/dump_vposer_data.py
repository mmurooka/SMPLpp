#!/usr/bin/env python3

# Ref. https://github.com/nghorbani/human_body_prior/blob/master/tutorials/vposer.ipynb

import sys
import os
import json
import torch
import numpy as np

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

def main(vposer_load_path, json_save_path):
    print("Load VPoser parameters from: ", os.path.abspath(vposer_load_path))
    vposer, _ = load_model(vposer_load_path,
                           model_code=VPoser,
                           remove_words_in_model_weights='vp_model.',
                           disable_grad=True)

    tensor_in = torch.rand(1, 32, dtype=torch.float32, requires_grad=True)
    tensor_out = vposer.decode(tensor_in)['pose_body']
    tensor_out_norm = tensor_out.norm()
    tensor_out_norm.backward()

    print("Save VPoser data to: ", os.path.abspath(json_save_path))
    data_json = {
        "VPoserDecoder.in": tensor_in.detach().numpy().tolist(),
        "VPoserDecoder.out": tensor_out.detach().numpy().tolist(),
        "VPoserDecoder.grad": tensor_in.grad.detach().numpy().tolist()
    }
    with open(json_save_path, "w+") as f:
        json.dump(data_json, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemError("Invalid number of arguments!\n"
                          "USAGE: {} <vposer_load_path> <json_save_path>".format(sys.argv[0]))
    main(sys.argv[1], sys.argv[2])
