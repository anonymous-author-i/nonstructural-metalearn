import numpy as np
import os
from sklearn.metrics import root_mean_squared_error as RMSE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

CASE = "contact"
title_list = [
    "First-Order",
    "L1-Adapt",
    "Meta-Adapt",
    "Meta-Adapt-FC",
    "Meta-LS-FC",
]
esti_plot_selected = [
    "First-Order",
    "L1-Adapt",
    "Meta-Adapt",
    "Meta-Adapt-FC",
]
esti_dir = "sim_contact_esti"

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm, colors

# Define Wrap-up function
from aux_module.quadrotor_visualize import Quadrotor_Visualize
import imageio

vis = Quadrotor_Visualize()
vis.bodyX_alpha = 0.8
vis.prop_alpha = 0.75
colorbar_max = 0


def load_data(traj_dir):
    traj_data = np.load(traj_dir)
    states = traj_data["x_real"]
    refs = traj_data["x_ref"]
    d_esti = traj_data["dx_pred"]
    d_real = traj_data["dx_real"]
    d_nomi = traj_data["dx_nomi"]
    return states, refs, d_esti, d_real, d_nomi


def get_statistics(traj_dir_list, verbose=0):
    state_list = []
    xref_list = []
    d_esti_list = []
    d_real_list = []
    d_nomi_list = []
    ctrl_errnorm_list = []
    esti_errnorm_list = []
    max_ctrlerr_list = []
    max_estierr_list = []
    case_id = 1
    for traj_dir in traj_dir_list:
        states, x_ref, d_esti, d_real, d_nomi = load_data(traj_dir)
        pos = states[0:3, :-1]
        pos_ref = x_ref[0:3, :]
        ctrl_errnorm = np.linalg.norm(pos - pos_ref, axis=0)
        esti_errnorm = np.linalg.norm(d_esti - d_real, axis=0)
        max_ctrlerr = np.max(ctrl_errnorm)
        max_estierr = np.max(esti_errnorm)
        rmse_ctrl = RMSE(pos, pos_ref)
        rmse_esti = RMSE(d_esti, d_real)
        if verbose == 1:
            print(f"{case_id} Control, RMSE: {rmse_ctrl}, Max Error: {max_ctrlerr}")
        elif verbose == 2:
            print(f"{case_id} Estimation, RMSE: {rmse_esti}, Max Error: {max_estierr}")
        max_ctrlerr_list.append(max_ctrlerr)
        max_estierr_list.append(max_estierr)
        state_list.append(states[:, :-1])
        xref_list.append(x_ref)
        d_esti_list.append(d_esti)
        d_real_list.append(d_real)
        d_nomi_list.append(d_nomi)
        ctrl_errnorm_list.append(ctrl_errnorm)
        esti_errnorm_list.append(esti_errnorm)

        case_id += 1

    return (
        state_list,
        xref_list,
        d_esti_list,
        d_real_list,
        d_nomi_list,
        ctrl_errnorm_list,
        esti_errnorm_list,
        max_ctrlerr_list,
        max_estierr_list,
    )


def plot_estimation(disturb_real_stack, disturb_pred_stack, fig_, title_4, legend_in_ax=True):
    plt.rcParams.update({"font.size": 16, "font.family": "calibri"})
    c1 = "lightcoral"
    c2 = "darkcyan"
    s1 = "solid"
    s2 = "dotted"
    lw = 2.5

    fig_.subplots_adjust(0.1, 0.12, 0.95, 0.9, 0.25, 0.3)
    ax_1 = fig_.add_subplot(3, 4, 1)
    ax_1.set_ylabel("$d_x$ [$m^2/s$]")
    ax_1.set_title(title_4[0])
    ax_1.grid()
    ax_1.plot(disturb_pred_stack[:, 0], color=c1, linestyle=s1, linewidth=lw)
    ax_1.plot(disturb_real_stack[:, 0], color=c2, linestyle=s2, linewidth=lw)
    ax_1.patch.set_color("gray")
    ax_1.patch.set_alpha(0.1)

    ax_2 = fig_.add_subplot(3, 4, 5)
    ax_2.set_ylabel("$d_y$ [$m^2/s$]")
    ax_2.grid()
    ax_2.plot(disturb_pred_stack[:, 1], color=c1, linestyle=s1, linewidth=lw)
    ax_2.plot(disturb_real_stack[:, 1], color=c2, linestyle=s2, linewidth=lw)
    ax_2.patch.set_color("gray")
    ax_2.patch.set_alpha(0.1)

    ax_3 = fig_.add_subplot(3, 4, 9)
    ax_3.set_ylabel("$d_z$ [$m^2/s$]")
    ax_3.grid()
    ax_3.plot(disturb_pred_stack[:, 2], color=c1, linestyle=s1, linewidth=lw)
    ax_3.plot(disturb_real_stack[:, 2], color=c2, linestyle=s2, linewidth=lw)
    ax_3.patch.set_color("gray")
    ax_3.patch.set_alpha(0.1)

    ax_4 = fig_.add_subplot(3, 4, 2)
    ax_4.set_title(title_4[1])
    ax_4.grid()
    ax_4.plot(disturb_pred_stack[:, 3], color=c1, linestyle=s1, linewidth=lw)
    ax_4.plot(disturb_real_stack[:, 3], color=c2, linestyle=s2, linewidth=lw)
    ax_4.patch.set_color("gray")
    ax_4.patch.set_alpha(0.1)

    ax_5 = fig_.add_subplot(3, 4, 6)
    ax_5.grid()
    ax_5.plot(disturb_pred_stack[:, 4], color=c1, linestyle=s1, linewidth=lw)
    ax_5.plot(disturb_real_stack[:, 4], color=c2, linestyle=s2, linewidth=lw)
    ax_5.patch.set_color("gray")
    ax_5.patch.set_alpha(0.1)

    ax_6 = fig_.add_subplot(3, 4, 10)
    ax_6.grid()
    ax_6.plot(disturb_pred_stack[:, 5], color=c1, linestyle=s1, linewidth=lw)
    ax_6.plot(disturb_real_stack[:, 5], color=c2, linestyle=s2, linewidth=lw)
    ax_6.patch.set_color("gray")
    ax_6.patch.set_alpha(0.1)

    ax_7 = fig_.add_subplot(3, 4, 3)
    ax_7.set_title(title_4[2])
    ax_7.grid()
    ax_7.plot(disturb_pred_stack[:, 6], color=c1, linestyle=s1, linewidth=lw)
    ax_7.plot(disturb_real_stack[:, 6], color=c2, linestyle=s2, linewidth=lw)
    ax_7.patch.set_color("gray")
    ax_7.patch.set_alpha(0.1)

    ax_8 = fig_.add_subplot(3, 4, 7)
    ax_8.grid()
    ax_8.plot(disturb_pred_stack[:, 7], color=c1, linestyle=s1, linewidth=lw)
    ax_8.plot(disturb_real_stack[:, 7], color=c2, linestyle=s2, linewidth=lw)
    ax_8.patch.set_color("gray")
    ax_8.patch.set_alpha(0.1)

    ax_9 = fig_.add_subplot(3, 4, 11)
    ax_9.grid()
    ax_9.plot(disturb_pred_stack[:, 8], color=c1, linestyle=s1, linewidth=lw)
    ax_9.plot(disturb_real_stack[:, 8], color=c2, linestyle=s2, linewidth=lw)
    ax_9.patch.set_color("gray")
    ax_9.patch.set_alpha(0.1)

    ax_10 = fig_.add_subplot(3, 4, 4)
    ax_10.set_title(title_4[3])
    ax_10.grid()
    ax_10.plot(disturb_pred_stack[:, 9], color=c1, linestyle=s1, linewidth=lw)
    ax_10.plot(disturb_real_stack[:, 9], color=c2, linestyle=s2, linewidth=lw)
    ax_10.patch.set_color("gray")
    ax_10.patch.set_alpha(0.1)

    ax_11 = fig_.add_subplot(3, 4, 8)
    ax_11.grid()
    ax_11.plot(disturb_pred_stack[:, 10], color=c1, linestyle=s1, linewidth=lw)
    ax_11.plot(disturb_real_stack[:, 10], color=c2, linestyle=s2, linewidth=lw)
    ax_11.patch.set_color("gray")
    ax_11.patch.set_alpha(0.1)

    ax_12 = fig_.add_subplot(3, 4, 12)
    ax_12.grid()
    ax_12.plot(disturb_pred_stack[:, 11], color=c1, linestyle=s1, linewidth=lw)
    ax_12.plot(disturb_real_stack[:, 11], color=c2, linestyle=s2, linewidth=lw)
    ax_12.patch.set_color("gray")
    ax_12.patch.set_alpha(0.1)

    legend_elements = [
        Line2D([0], [0], linestyle=s1, color=c1, lw=2, label="Estimation with Baseline"),
        Line2D([0], [0], linestyle=s2, color=c2, lw=2, label="Truth"),
    ]

    # fig_.legend(handles=legend_elements, loc="lower center", ncol=3, framealpha=1)
    if legend_in_ax: ax_10.legend(handles=legend_elements)


def plot_error_boxplot(df_esti, figsize, save_path, order=None):
    # Prepare the data for plotting
    sns.set_theme("paper", style="ticks", font_scale=1.5)
    # Set font properties
    # plt.rcParams.update({"font.size": 14, "font.family": "calibri"})

    # Plot the boxplot with dual y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df_esti,
        x="Method",
        y="Estimation Error",
        ax=ax1,
        color="lightblue",
        width=0.5,
        linewidth=1.2,
        order=order,
    )
    # Set labels and grid
    ax1.set_ylabel("Estimation (w. Baseline) Error [$m/s^2$]", color="black")
    ax1.set_xlabel(None)
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(visible=True, linestyle="--", alpha=0.5)

    # Add a legend
    legend_elements = [
        Line2D([0], [0], color="lightblue", lw=4, label="Estimation Error"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", framealpha=1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)


if __name__ == "__main__":
    """Plot Estimation/Prediction Results"""
    fig_esti = plt.figure(figsize=(11, 7))

    # Get sim esti trajectories
    print("Loading Estimation Cases.")
    traj_dir_list = [
        esti_dir+'/'+file for file in os.listdir("{}".format(esti_dir)) if file.endswith(".npz")
    ]
    (
        state_list,
        xref_list,
        d_esti_list,
        d_real_list,
        d_nomi_list,
        ctrl_errnorm_list,
        esti_errnorm_list,
        max_ctrlerr_list,
        max_estierr_list,
    ) = get_statistics(traj_dir_list, verbose=2)
    disturb_real_stack = []
    disturb_pred_stack = []
    title_4 = []
    for i, title in list(zip(range(len(traj_dir_list)), title_list)):
        if title in esti_plot_selected:
            dx_real_seq = d_real_list[i]
            dx_nomi_seq = d_nomi_list[i]
            dx_pred_seq = d_esti_list[i]
            # Compute disturbance
            disturb_real_seq = dx_real_seq - dx_nomi_seq
            disturb_pred_seq = dx_pred_seq - dx_nomi_seq
            # Stack disturbances
            disturb_real_stack += [disturb_real_seq[3:6]]
            disturb_pred_stack += [disturb_pred_seq[3:6]]
            # Add to list
            if title == "L1-Adapt":
                title = "$\mathcal{L}_1$-Adapt"
            title_4 += [title]

    disturb_real_stack = np.vstack(disturb_real_stack)
    disturb_pred_stack = np.vstack(disturb_pred_stack)

    plot_estimation(disturb_real_stack.T, disturb_pred_stack.T, fig_esti, title_4)
    fig_esti.savefig("img/{0}_estimation.png".format(CASE), dpi=300)

    """Plot Estimation Error Boxplot"""
    # Prepare the data for plotting
    data_dict = {
            "Method": sum(
                [
                    [title_list[i]] * state_list[i].shape[1]
                    for i in range(len(title_list))
                ],
                [],
            ),
            "Estimation Error": list(np.hstack(esti_errnorm_list)),
            "RMSE Estimation": sum(
                [
                    [RMSE(d_esti, d_real)] * state_list[i].shape[1]
                    for i, d_esti, d_real in zip(
                        range(len(title_list)), d_esti_list, d_real_list
                    )
                ],
                [],
            ),
            "Max Estimation Error": sum(
                [
                    [max_estierr_list[i]] * state_list[i].shape[1]
                    for i in range(len(title_list))
                ],
                [],
            ),
        }
    df_esti = pd.DataFrame(data_dict)
    plot_error_boxplot(
        df_esti,
        figsize=(8, 4),
        save_path="img/{0}_error_boxplot.png".format(CASE),
        order=title_list,
    )

    """Make Estimation Gif"""
    def create_estimation_gif(disturb_real_stack, disturb_pred_stack, title_4, save_path="img/estimation.gif"):
        plt.rcParams.update({"font.size": 16, "font.family": "calibri"})
        c1 = "lightcoral"
        c2 = "darkcyan"
        s1 = "solid"
        s2 = "dotted"
        lw = 2.5

        frames = []
        num_steps = disturb_pred_stack.shape[0]
        skip=5

        for step in range(0, num_steps, skip):
            # fig, axes = plt.subplots(3, 4, figsize=(12, 7))
            # fig.subplots_adjust(0.1, 0.12, 0.95, 0.9, 0.25, 0.3)

            # for i, ax in enumerate(axes.flatten()):
            #     if i >= disturb_real_stack.shape[1]:
            #         ax.axis("off")
            #         continue

            #     ax.grid()
            #     ax.plot(disturb_pred_stack[:step + 1, i], color=c1, linestyle=s1, linewidth=lw, label="Estimation")
            #     ax.plot(disturb_real_stack[:step + 1, i], color=c2, linestyle=s2, linewidth=lw, label="Truth")
            #     ax.patch.set_color("gray")
            #     ax.patch.set_alpha(0.1)

            #     if i < len(title_4):
            #         ax.set_title(title_4[i])

            fig = plt.figure(figsize=(12, 7))
            plot_estimation(disturb_real_stack[:step + 1, :], disturb_pred_stack[:step + 1, :], fig, title_4, False)
            
            legend_elements = [
                Line2D([0], [0], linestyle=s1, color=c1, lw=2, label="Estimation with Baseline"),
                Line2D([0], [0], linestyle=s2, color=c2, lw=2, label="Truth"),
            ]
            fig.legend(handles=legend_elements, loc="lower center", ncols=3)

            # Save the current frame
            frame_path = f"img/temp_frame_{step}.png"
            fig.savefig(frame_path, dpi=100)
            frames.append(imageio.imread(frame_path))
            plt.close(fig)

        # Create the GIF
        imageio.mimsave(save_path, frames, fps=60)

        # Clean up temporary frames
        for step in range(0, num_steps, skip):
            os.remove(f"img/temp_frame_{step}.png")


    # Call the function to create the GIF
    create_estimation_gif(disturb_real_stack.T, disturb_pred_stack.T, title_4)