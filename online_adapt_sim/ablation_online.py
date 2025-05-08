import numpy as np
import argparse
import json
import os
from multiprocessing import Pool

from sklearn.metrics import root_mean_squared_error as RMSE
from dynamics.model_nomi import Quadrotor
from model_wrapper import *
from estimator.meta_adaptive import *
from utils import *

import matplotlib.pyplot as plt


def single_rollout(
    x_traj, u_traj, dx_real_seq, dx_nomi_seq, estimator, settings, mode="adaptive"
):
    Theta_adapt = np.zeros(
        len(settings["output_select"]) * settings["hidden_sizes"][-1]
    )
    z_hat = np.zeros(len(settings["output_select"]))
    d_hat = np.zeros(len(settings["output_select"]))

    ff_input_init = np.vstack(
        [
            x_traj[settings["state_select"], 0:1],
            u_traj[settings["control_select"], 0:1],
        ]
    )
    estimator.init_learner_and_stack(
        init_ffinput=ff_input_init, init_dx=dx_real_seq[:, 0:1]
    )

    dx_pred_seq = np.zeros_like(dx_real_seq)
    for idx in range(x_traj.shape[1]):
        x = x_traj[:, idx]
        u = u_traj[:, idx]
        dx_nomi_i = dx_nomi_seq[:, idx]
        dx_real_i = dx_real_seq[:, idx]
        disturb_i = dx_real_i - dx_nomi_i
        disturb_i = disturb_i[settings["output_select"]].reshape((-1, 1))
        ff_input_i = np.vstack(
            [
                x_traj[settings["state_select"], idx : idx + 1],
                u_traj[settings["control_select"], idx : idx + 1],
            ]
        )

        ff_inputs = estimator.ff_inputs
        dx_pred_i = dx_nomi_i.copy()

        estimator.store_data(ff_input_i, np.expand_dims(dx_real_i, axis=1), disturb_i)

        # online learning
        if mode == "adaptive":
            Theta_adapt1 = estimator.concurrent_learn(Theta_adapt)
        elif mode == "direct":
            Theta_adapt1 = estimator.direct_least_square(reg=0.01)
        else:
            raise ValueError("Invalid mode. Choose 'adaptive' or 'direct'.")

        # Theta_adapt = learn_esti.direct_least_square()
        d_hat = estimator.ff_model.predictor.predict(ff_inputs, Theta_adapt)

        # online: forward estimation
        z_hat1, d_hat = estimator.d_esti_dx(z_hat, dx_real_i, x, u, Theta_adapt)
        dx_pred_i[settings["output_select"]] = (
            dx_nomi_i[settings["output_select"]] + d_hat
        )

        dx_pred_seq[:, idx] = dx_pred_i

        # update
        Theta_adapt = Theta_adapt1
        z_hat = z_hat1

    return dx_pred_seq


def go_sim_in_batch(
    x_seq_batch,
    u_seq_batch,
    dx_real_seq_batch,
    dx_nomi_seq_batch,
    estimator,
    settings,
    mode="adaptive",
    process_use=1,
):

    dx_pred_batch = []

    input_list = [
        (
            x_seq_batch[:, :, ii],
            u_seq_batch[:, :, ii],
            dx_real_seq_batch[:, :, ii],
            dx_nomi_seq_batch[:, :, ii],
            estimator,
            settings,
            mode,
        )
        for ii in range(x_seq_batch.shape[2])
    ]
    with Pool(process_use) as pool:
        dx_pred_batch = pool.starmap(single_rollout, input_list)

    dx_pred_batch = np.dstack(dx_pred_batch)
    # print(dx_nomi_seq_batch.shape, dx_pred_batch.shape)

    return dx_pred_batch


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
        default="meta_mlpt_h0_bl5",
        help="name of the learned encoder model",
    )
    parser.add_argument(
        "--trajdata_dir",
        type=str,
        default="datasets",
        help="dir for simulated data usage",
    )
    parser.add_argument(
        "--discrete_h",
        type=float,
        default=0.02,
        help="discrete time step for simulation",
    )
    parser.add_argument(
        "--process_use",
        type=int,
        default=4,
        help="number of cpu cores for ablation study",
    )
    parser.add_argument("--device", default="cpu", help="device used for training")

    opt = parser.parse_args()

    # load setups
    setups = {}
    with open(
        "learned_models/" + opt.model_name + "/model_setup.json", "r", encoding="utf-8"
    ) as f:
        setups = json.load(f)

    # get predictors for direct learning
    model_torch = get_predictor_model(opt, setups)
    pth_path = "learned_models/" + opt.model_name + "/model_param.pth"
    model_torch.load_state_dict(torch.load(pth_path))
    model_torch.eval()

    # initialize dynamics
    nomi_dyn = Quadrotor()

    # initialize online estimators
    settings = setups[opt.encoder]
    predictor = PredictorShell_Adapt(settings, torch_model=model_torch, mode="normal")
    model_adapt = Model_Wrapper(
        nominal_model=nomi_dyn, predictor_model=predictor, discrete_h=opt.discrete_h
    )
    learn_esti_adaptive = MetaAdaptiveEstimator(
        ff_model=model_adapt,
        ff_enable=True,
        adapt_gain=5.0
        * np.eye(len(settings["output_select"]) * settings["hidden_sizes"][-1]),
        feedback_gain=0.0,
        concurrent_learn_len=5,
    )
    learn_esti_direct = MetaAdaptiveEstimator(
        ff_model=model_adapt,
        ff_enable=True,
        adapt_gain=5.0
        * np.eye(len(settings["output_select"]) * settings["hidden_sizes"][-1]),
        feedback_gain=0.0,
        concurrent_learn_len=settings["baselearn_len"] + 1,
    )
    # load simulated trajectories and go online learning
    datasets_dir = opt.trajdata_dir
    folders = [
        folder
        for folder in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, folder))
    ]

    # Make array dict for saving
    array_dict = {}

    # Go through all folders (datasets)
    for folder in folders:
        raw_data_dir = os.path.join(datasets_dir, folder)
        print("Loading data from: ", raw_data_dir)
        traj_dir_list = [
            file for file in os.listdir(raw_data_dir) if file.endswith(".npz")
        ]
        print("Number of trajectories: ", len(traj_dir_list))

        dx_real_set = []
        dx_pred_set_adaptive = []
        dx_pred_set_direct = []

        # Go through all batches in the folder
        for i, traj_dir in enumerate(traj_dir_list):
            traj_i = i + 1
            print("progress: {0}/{1}".format(traj_i, len(traj_dir_list)))
            data_ = np.load(raw_data_dir + "/" + traj_dir)
            x_seq_batch = data_["x_seq_batch"]
            u_seq_batch = data_["u_seq_batch"]
            # print(x_seq_batch.shape, u_seq_batch.shape)
            dx_real_seq_batch = data_["dx_real_seq_batch"]
            dx_nomi_seq_batch = data_["dx_nomi_seq_batch"]
            output_seq_batch = dx_real_seq_batch - dx_nomi_seq_batch

            dx_real_set += [dx_real_seq_batch.flatten().reshape((-1, 1))]
            dx_pred_batch = go_sim_in_batch(
                x_seq_batch,
                u_seq_batch,
                dx_real_seq_batch,
                dx_nomi_seq_batch,
                learn_esti_adaptive,
                settings,
                mode="adaptive",
                process_use=opt.process_use
            )
            dx_pred_set_adaptive += [dx_pred_batch.flatten().reshape((-1, 1))]
            dx_pred_batch = go_sim_in_batch(
                x_seq_batch,
                u_seq_batch,
                dx_real_seq_batch,
                dx_nomi_seq_batch,
                learn_esti_direct,
                settings,
                mode="direct",
                process_use=opt.process_use
            )
            dx_pred_set_direct += [dx_pred_batch.flatten().reshape((-1, 1))]

        dx_real_array = np.concatenate(dx_real_set, axis=0)
        dx_pred_adaptive_array = np.concatenate(dx_pred_set_adaptive, axis=0)
        dx_pred_direct_array = np.concatenate(dx_pred_set_direct, axis=0)
        
        # with torch.no_grad():
        #     dx_real_batches = torch.tensor(dx_real_batches, dtype=torch.float32)
        #     dx_pred_adaptive_batches = torch.tensor(dx_pred_adaptive_batches, dtype=torch.float32)
        #     dx_pred_direct_batches = torch.tensor(dx_pred_direct_batches, dtype=torch.float32)
        #     mse_adaptive = torch.nn.MSELoss(reduction="mean")(
        #         dx_real_batches, dx_pred_adaptive_batches
        #     ).item()
        #     mse_direct = torch.nn.MSELoss(reduction="mean")(
        #         dx_real_batches, dx_pred_direct_batches
        #     ).item()

        # array_dict[f"{folder}"] = {
        #     "mse_adaptive":  mse_adaptive,
        #     "mse_direct": mse_direct,
        # }

        array_dict[f"{folder}"] = {
            "dx_real_batches": dx_real_array,
            "dx_pred_adaptive_batches": dx_pred_adaptive_array,
            "dx_pred_direct_batches": dx_pred_direct_array,
        }

    # Save results
    save_dir = "./ablation"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    np.savez(save_dir + "/{}.npz".format(opt.model_name), **array_dict)
