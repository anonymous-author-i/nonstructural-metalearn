import numpy as np
import os
import json
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm, colors
from matplotlib import rcParams

from predictors.encoders import *


def get_predictor_model(opt, setups):
    settings = setups[opt.encoder]
    if opt.encoder == "MLP":
        model = MLP_Encoder(
            state_dim=len(settings["state_select"]),
            control_dim=len(settings["control_select"]),
            output_dim=len(settings["output_select"]),
            hidden_sizes=settings["hidden_sizes"],
            dropout=settings["drop_out"],
            device=opt.device,
        )
        model.to(opt.device)
        count_param(model)
        return model

    elif opt.encoder == "MLPTimeEmbed":
        model = MLPTimeEmbed_Encoder(
            state_dim=len(settings["state_select"]),
            control_dim=len(settings["control_select"]),
            output_dim=len(settings["output_select"]),
            history_len=settings["history_len"],
            hidden_sizes=settings["hidden_sizes"],
            dropout=settings["drop_out"],
            device=opt.device,
        )
        model.to(opt.device)
        count_param(model)
        return model
    
    elif opt.encoder == "MetaMLP":
        model = MetaMLP_Basis(
            state_dim=len(settings["state_select"]),
            control_dim=len(settings["control_select"]),
            output_dim=len(settings["output_select"]),
            hidden_sizes=settings["hidden_sizes"],
            dropout=settings["drop_out"],
            device=opt.device,
        )
        model.to(opt.device)
        count_param(model)
        return model

    elif opt.encoder == "MetaMLPTimeEmbed":
        model = MetaMLPTimeEmbed_Basis(
            state_dim=len(settings["state_select"]),
            control_dim=len(settings["control_select"]),
            output_dim=len(settings["output_select"]),
            history_len=settings["history_len"],
            hidden_sizes=settings["hidden_sizes"],
            dropout=settings["drop_out"],
            device=opt.device,
        )
        model.to(opt.device)
        count_param(model)
        return model
    
    # elif opt.encoder == "MetaMLPtMLP":
    #     settings = setups[opt.encoder]
    #     model = MetaMLPtMLP_Basis(
    #         state_dim=len(settings["state_select"]),
    #         control_dim=len(settings["control_select"]),
    #         output_dim=len(settings["output_select"]),
    #         base_reg=settings["base_reg"],
    #         history_len=settings["history_len"],
    #         baselearn_len=settings["baselearn_len"],
    #         future_horizon=settings["future_horizon"],
    #         mlpt_encoder_sizes=settings["mlpt_encoder_sizes"],
    #         mlpt_encoded_dim=settings["mlpt_encoded_dim"],
    #         hidden_sizes=settings["hidden_sizes"],
    #         dropout=settings["drop_out"],
    #         device=opt.device,
    #     )
    #     model.to(opt.device)


def count_param(model):
    # count params
    num_params = 0
    params = model.parameters()
    for param in params:
        num_params += torch.prod(torch.tensor(param.size()))

    print("total params: ", num_params)
