import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from sklearn.metrics import mean_squared_error as MSE

def get_model_settings(model_name, dir_="learned_models"):
    """
    Get model settings based on the model name.
    """
    file_dir = os.path.join(dir_, model_name)
    if os.path.exists(file_dir):
        # Load setups
        setups = {}
        with open(
            dir_ + "/" + model_name + "/model_setup.json", "r", encoding="utf-8"
        ) as f:
            setups = json.load(f)
        settings = setups[list(setups.keys())[0]]
    return settings

if __name__ == "__main__":
    # Load data
    data_dir = "./ablation"
    file_list = [
        file for file in os.listdir(data_dir) if file.endswith(".npz")
    ]
    print("Number of Models: ", len(file_list))

    df_ablation = {
        "Dataset": [],
        "MSE": [],
        # "Prediction_Error": [],
        "History_len": [],
        "Baselearn_len": [],
        "Mode": [],
    }
    for f_ in file_list:
        data_ = np.load(data_dir + "/" + f_, allow_pickle=True)
        settings = get_model_settings(f_[:-4])
        baselearn_len = settings["baselearn_len"]
        history_len = settings["history_len"]
        for key in data_.keys():
            data_dict = data_[key].item()

            # # Compute error norm
            # pred_error_adaptive = list(
            #     np.linalg.norm(
            #         data_dict["dx_real_array"].reshape((3, -1)) - data_dict["dx_pred_adaptive_array"].reshape((3, -1)), axis=0
            #     )
            # )
            # pred_error_direct = list(
            #     np.linalg.norm(
            #         data_dict["dx_real_array"].reshape((3, -1)) - data_dict["dx_pred_direct_array"].reshape((3, -1)), axis=0
            #     )
            # )
            # mse_adaptive = MSE(
            #     data_dict["dx_real_array"], data_dict["dx_pred_adaptive_array"])
            # mse_direct = MSE(
            #     data_dict["dx_real_array"], data_dict["dx_pred_direct_array"])
            
            # # Make dictionary
            # df_ablation["Dataset"] += [key] * len(pred_error_adaptive)
            # df_ablation["Prediction_Error"] += pred_error_adaptive
            # df_ablation["MSE"] += [mse_adaptive] * len(pred_error_adaptive)
            # df_ablation["Mode"] += ["Adaptive"] * len(pred_error_adaptive)
            # df_ablation["History_len"] += [history_len] * len(pred_error_adaptive)
            # df_ablation["Baselearn_len"] += [baselearn_len] * len(pred_error_adaptive)
            
            # df_ablation["Dataset"] += [key] * len(pred_error_direct)
            # df_ablation["Prediction_Error"] += pred_error_direct
            # df_ablation["MSE"] += [mse_direct] * len(pred_error_direct)
            # df_ablation["Mode"] += ["Direct"] * len(pred_error_direct)
            # df_ablation["History_len"] += [history_len] * len(pred_error_direct)
            # df_ablation["Baselearn_len"] += [baselearn_len] * len(pred_error_direct)

            mse_adaptive = MSE(
                data_dict["dx_real_array"], data_dict["dx_pred_adaptive_array"])
            mse_direct = MSE(
                data_dict["dx_real_array"], data_dict["dx_pred_direct_array"])

            df_ablation["Dataset"] += [key]
            df_ablation["MSE"] += [mse_adaptive]
            df_ablation["Mode"] += ["Adaptive"]
            df_ablation["History_len"] += [history_len]
            df_ablation["Baselearn_len"] += [baselearn_len]
            
            df_ablation["Dataset"] += [key]
            df_ablation["MSE"] += [mse_direct]
            df_ablation["Mode"] += ["Direct"]
            df_ablation["History_len"] += [history_len]
            df_ablation["Baselearn_len"] += [baselearn_len]

    df_ablation = pd.DataFrame(df_ablation)
    print(df_ablation)

    # make heatmap of direct mode with 2 datasets
    fig1 = plt.figure(figsize=(12, 5))
    ax1, ax2 = fig1.subplots(1, 2)
    fig1.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.3)
    sns.set_theme("paper", font_scale=1.8, font="calibri")

    df_direct = df_ablation[df_ablation["Mode"] == "Direct"]
    df_ds1 = df_direct[df_direct["Dataset"] == "meta-learn"]
    df_ds2 = df_direct[df_direct["Dataset"] == "out-of-distribution"]
    sns.heatmap(
        df_ds1.pivot_table(index="History_len", columns="Baselearn_len", values="MSE"),
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        ax=ax1,
    )
    sns.heatmap(
        df_ds2.pivot_table(index="History_len", columns="Baselearn_len", values="MSE"),
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        ax=ax2,
    )
    ax1.set_title("(a) meta-learn MSE Loss", fontsize=15, font="calibri")
    ax1.set_xlabel("Regression Length", fontsize=15, font="calibri")
    ax1.set_ylabel("Time-Delay Length", fontsize=15, font="calibri")
    ax2.set_title("(b) out-of-distribution MSE Loss", fontsize=15, font="calibri")
    ax2.set_xlabel("Regression Length", fontsize=15, font="calibri")
    ax2.set_ylabel("Time-Delay Length", fontsize=15, font="calibri")
    fig1.savefig("img/ablation_heatmap.png", dpi=300, bbox_inches="tight")

    # make lineplot of adaptive mode with history_len=0
    df_adapt = df_ablation[df_ablation["Mode"] == "Adaptive"]
    df_adapt = df_adapt[df_adapt["History_len"] == 0]

    fig4 = plt.figure(figsize=(4, 4))
    sns.set_theme("paper", style="darkgrid", font_scale=1.4, font="calibri")
    ax7 = fig4.add_subplot(111)
    sns.lineplot(
        data=df_adapt,
        x="Baselearn_len",
        y="MSE",
        hue="Dataset",
        style="Dataset",
        markers=True,
        dashes=False,
        ax=ax7,
        markersize=5,  # Increase marker size
        linewidth=2,  # Thicker lines
    )
    ax7.set_xticks(df_adapt["Baselearn_len"].unique())
    ax7.set_title("(c) Online Parameter Estimation", font="calibri")
    ax7.set_xlabel("Baselearn Length", font="calibri")
    ax7.set_ylabel("Adaptive MSE Loss", font="calibri")

    fig4.savefig("img/ablation_adaptive_h0.png", dpi=300, bbox_inches="tight")