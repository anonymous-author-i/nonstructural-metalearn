import torch
from torch.nn import init
import numpy as np
import argparse
import json

from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        default="MetaMLP",
        help="choose an encoder model for prediction",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta_mlp",
        help="the current run case name",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="data/metatest",
        help="directory of raw simulated data",
    )
    parser.add_argument(
        "--trajectory_clip",
        type=int,
        default=100,
        help="clip the trajectories with limited length < minimal trajectory length",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=200,
        help="minibatch_size used for training",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1e-6,
        help="l2 regulation of meta-learner",
    )

    parser.add_argument("--device", default="cuda", help="device used for training")

    opt = parser.parse_args()

    # load setups
    setups = {}
    with open(
        "learned_models/" + opt.model_name + "/model_setup.json", "r", encoding="utf-8"
    ) as f:
        setups = json.load(f)

    # get dataloader
    meta_test_testing, _, _ = get_dataloader(
        opt, setups, split_ratio=[10, 0, 0], mode="test"
    )

    # get model
    model = get_encoder_model(opt, setups, mode="test")
    pth_path = "learned_models/" + opt.model_name + "/model_param.pth"
    model.load_state_dict(torch.load(pth_path))
    # # nominal case: zero all parameters
    # for key in model.state_dict():
    #     if key.split('.')[-1] == 'weight':
    #         if 'conv' in key:
    #             init.zeros_(model.state_dict()[key])
    #         if 'bn' in key:
    #             model.state_dict()[key][...] = 0
    #     elif key.split('.')[-1] == 'bias':
    #         model.state_dict()[key][...] = 0

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_labels in meta_test_testing:
            loss_mse, _ = regression_loss(
                model,
                batch_inputs,
                batch_labels,
                param_reg=opt.reg,
                device=opt.device,
            )
            test_loss += loss_mse.item()

    print(opt.raw_data_dir[7:], " ", test_loss / len(meta_test_testing))
