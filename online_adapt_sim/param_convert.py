import torch
import numpy as np
import argparse
import json
from utils import get_predictor_model
from scipy.io import savemat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        default="MetaMLPTimeEmbed",
        help="choose a direct model for prediction",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta_mlpt_h0_bl15",
        help="name of the learned encoder model",
    )
    parser.add_argument("--device", default="cpu", help="device used for training")

    opt = parser.parse_args()

    setups = {}
    with open(
        "learned_models/" + opt.model_name + "/model_setup.json", "r", encoding="utf-8"
    ) as f:
        setups = json.load(f)

    # Get predictors for direct learning
    model_torch = get_predictor_model(opt, setups)
    pth_path = "learned_models/" + opt.model_name + "/model_param.pth"
    model_torch.load_state_dict(torch.load(pth_path))
    model_torch.eval()

    # Save the model parameters
    params_dict = {}
    idx = 1
    for key, value in model_torch.state_dict().items():
        param_cpu = value.cpu()
        param_array = param_cpu.detach().numpy()
        params_dict["param_{}".format(idx)] = param_array
        idx += 1

    savemat("learned_models/" + opt.model_name + "/model_param.mat", 
            params_dict)
    print("Model parameters saved to model_param.mat")