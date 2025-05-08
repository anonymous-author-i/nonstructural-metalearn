import torch
import numpy as np
import argparse
import json
from utils import get_encoder_model
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
        default="meta_mlpt_his0_bl20",
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
    model_torch = get_encoder_model(opt, setups, mode="test")
    pth_path = "learned_models/" + opt.model_name + "/model_param.pth"
    model_torch.load_state_dict(torch.load(pth_path))
    model_torch.eval()

    # Pytorch model parameter to a vector
    params = []
    for param in model_torch.parameters():
        param_cpu = param.cpu()
        param_array = param_cpu.detach().numpy().reshape(-1, 1)
        params += [param_array]
    params = np.vstack(params)

    # Save the model parameters
    savemat("learned_models/" + opt.model_name + "/model_param.mat", {"params": params})
    print("Model parameters saved to model_param.mat")