import numpy as np
import os
from sklearn.metrics import root_mean_squared_error as RMSE

CASE = "lemniscate"
title_list = [
    "Baseline",
    "First-Order",
    "Vanilla-NN",
    "L1-Adapt",
    "Meta-Adapt",
    "Meta-Adapt-FC",
    "Meta-LS-FC",
]
esti_plot_selected = [
    "Vanilla-NN",
    "L1-Adapt",
    "Meta-Adapt",
    "Meta-Adapt-FC",
]
ctrl_dir = "sim_lemniscate_ctrl"

esti_dir = "sim_lemniscate_esti"

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm, colors

# Define Wrap-up function
from aux_module.quadrotor_visualize import Quadrotor_Visualize

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
        state_list.append(states)
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


def addsubplot_traj(xk_seq, refk_seq, errnorm_list, max_v, ax, title, start, end, stp):
    plt.rcParams.update({"font.size": 16, "font.family": "calibri"})
    ax.set_title(title, pad=-5, fontsize=16)
    ax.view_init(elev=24, azim=77)
    ax.set_box_aspect((6, 6, 2.5))
    ax.invert_zaxis()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    # ax2.set_xlim([-4, 4])
    # ax2.set_ylim([-4, 4])
    # ax.set_zlim([0, -1])
    # ax.set_zticks([-1.0, -0.5, 0.0])
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.plot(
        refk_seq[start:end, 0],
        refk_seq[start:end, 1],
        refk_seq[start:end, 2],
        "k",
        linestyle="-.",
    )
    for lossk, x in zip(errnorm_list[start:end:stp], xk_seq[start:end:stp]):
        v = (lossk - 0.01) / (max_v - 0.01)
        color = tuple(cm.turbo(v))
        if title == "Baseline":
            color = "gray"
        vis.plot_quadrotorEul(ax=ax, x=x[0:12], pc_list=[color] * 4, bc="k")


def plot_states(state_real, state_desire, fig_, title):
    plt.rcParams.update({"font.size": 16, "font.family": "calibri"})
    c1 = "lightcoral"
    c2 = "darkcyan"
    s1 = "solid"
    s2 = "dotted"
    lw = 3

    fig_.subplots_adjust(0.1, 0.12, 0.95, 0.9, 0.5, 0.3)
    ax_1 = fig_.add_subplot(3, 4, 1)
    # x2_1.set_xlabel("t")
    ax_1.set_ylabel("Pos x")
    ax_1.grid()
    ax_1.plot(state_real[:, 0], color=c1, linestyle=s1, linewidth=lw)
    ax_1.plot(state_desire[:, 0], color=c2, linestyle=s2, linewidth=lw)

    ax_2 = fig_.add_subplot(3, 4, 5)
    # ax2_2.set_xlabel("t")
    ax_2.set_ylabel("Pos y")
    ax_2.grid()
    ax_2.plot(state_real[:, 1], color=c1, linestyle=s1, linewidth=lw)
    ax_2.plot(state_desire[:, 1], color=c2, linestyle=s2, linewidth=lw)

    ax_3 = fig_.add_subplot(3, 4, 9)
    # ax2_3.set_xlabel("t")
    ax_3.set_ylabel("Pos z")
    ax_3.grid()
    ax_3.plot(state_real[:, 2], color=c1, linestyle=s1, linewidth=lw)
    ax_3.plot(state_desire[:, 2], color=c2, linestyle=s2, linewidth=lw)

    ax_4 = fig_.add_subplot(3, 4, 2)
    # x2_4.set_xlabel("t")
    ax_4.set_ylabel("Vel x")
    ax_4.grid()
    ax_4.plot(state_real[:, 3], color=c1, linestyle=s1, linewidth=lw)
    ax_4.plot(state_desire[:, 3], color=c2, linestyle=s2, linewidth=lw)

    ax_5 = fig_.add_subplot(3, 4, 6)
    # ax2_5.set_xlabel("t")
    ax_5.set_ylabel("Vel y")
    ax_5.grid()
    ax_5.plot(state_real[:, 4], color=c1, linestyle=s1, linewidth=lw)
    ax_5.plot(state_desire[:, 4], color=c2, linestyle=s2, linewidth=lw)

    ax_6 = fig_.add_subplot(3, 4, 10)
    # ax2_6.set_xlabel("t")
    ax_6.set_ylabel("Vel z")
    ax_6.grid()
    ax_6.plot(state_real[:, 5], color=c1, linestyle=s1, linewidth=lw)
    ax_6.plot(state_desire[:, 5], color=c2, linestyle=s2, linewidth=lw)

    ax_7 = fig_.add_subplot(3, 4, 3)
    # ax2_7.set_xlabel("t")
    ax_7.set_ylabel("Pitch")
    ax_7.grid()
    ax_7.plot(state_real[:, 6], color=c1, linestyle=s1, linewidth=lw)
    ax_7.plot(state_desire[:, 6], color=c2, linestyle=s2, linewidth=lw)

    ax_8 = fig_.add_subplot(3, 4, 7)
    # ax2_8.set_xlabel("t")
    ax_8.set_ylabel("Pitch")
    ax_8.grid()
    ax_8.plot(state_real[:, 7], color=c1, linestyle=s1, linewidth=lw)
    ax_8.plot(state_desire[:, 7], color=c2, linestyle=s2, linewidth=lw)

    ax_9 = fig_.add_subplot(3, 4, 11)
    # ax2_9.set_xlabel("t")
    ax_9.set_ylabel("Yaw")
    ax_9.grid()
    ax_9.plot(state_real[:, 8], color=c1, linestyle=s1, linewidth=lw)
    ax_9.plot(state_desire[:, 8], color=c2, linestyle=s2, linewidth=lw)

    ax_10 = fig_.add_subplot(3, 4, 4)
    # ax2_10.set_xlabel("t")
    ax_10.set_ylabel("Omega x")
    ax_10.grid()
    ax_10.plot(state_real[:, 9], color=c1, linestyle=s1, linewidth=lw)
    ax_10.plot(state_desire[:, 9], color=c2, linestyle=s2, linewidth=lw)

    ax_11 = fig_.add_subplot(3, 4, 8)
    # ax2_8.set_xlabel("t")
    ax_11.set_ylabel("Omega y")
    ax_11.grid()
    ax_11.plot(state_real[:, 10], color=c1, linestyle=s1, linewidth=lw)
    ax_11.plot(state_desire[:, 10], color=c2, linestyle=s2, linewidth=lw)

    ax_12 = fig_.add_subplot(3, 4, 12)
    # ax2_9.set_xlabel("t")
    ax_12.set_ylabel("Omega z")
    ax_12.grid()
    ax_12.plot(state_real[:, 11], color=c1, linestyle=s1, linewidth=lw)
    ax_12.plot(state_desire[:, 11], color=c2, linestyle=s2, linewidth=lw)

    legend_elements = [
        Line2D([0], [0], linestyle=s1, color=c1, lw=2, label="Tracking"),
        Line2D([0], [0], linestyle=s2, color=c2, lw=2, label="Reference"),
    ]

    fig_.legend(handles=legend_elements, loc="lower center", ncol=3, framealpha=1)

    fig_.savefig("img/x_plot_{}.png".format(title), dpi=300)


def plot_estimation(disturb_real_stack, disturb_pred_stack, fig_, title_4):
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
        Line2D([0], [0], linestyle=s1, color=c1, lw=2, label="Estimation"),
        Line2D([0], [0], linestyle=s2, color=c2, lw=2, label="Truth"),
    ]

    # fig_.legend(handles=legend_elements, loc="lower center", ncol=3, framealpha=1)
    ax_10.legend(handles=legend_elements)

    fig_.savefig("img/{0}_estimation.png".format(CASE), dpi=300)


if __name__ == "__main__":
    """Plot 3D Trajectories"""
    fig1 = plt.figure(figsize=(15, 10))
    fig1.subplots_adjust(
        left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.0
    )
    ax11 = fig1.add_subplot(241, projection="3d")
    ax12 = fig1.add_subplot(242, projection="3d")
    ax13 = fig1.add_subplot(243, projection="3d")
    ax14 = fig1.add_subplot(244, projection="3d")
    ax21 = fig1.add_subplot(245, projection="3d")
    ax22 = fig1.add_subplot(246, projection="3d")
    ax23 = fig1.add_subplot(247, projection="3d")
    ax24 = fig1.add_subplot(248, projection="3d")
    ax_list = [ax11, ax12, ax13, ax14, ax21, ax22, ax23, ax24]

    # Get sim control trajectories
    print("Loading Control Cases.")
    traj_dir_list = [
        ctrl_dir+'/'+file for file in os.listdir("{}".format(ctrl_dir)) if file.endswith(".npz")
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
    ) = get_statistics(traj_dir_list, verbose=1)
    max_colorbar1 = max(max_ctrlerr_list[1:]) + 0.01  # exclude baseline
    for i, title, ax in zip(range(len(traj_dir_list)), title_list, ax_list):
        x_real_seq = state_list[i]
        pos_reference = xref_list[i][0:3, :]
        errnorm_list = ctrl_errnorm_list[i]
        if title == "L1-Adapt":
            title = "$\mathcal{L}_1$-Adapt"
        addsubplot_traj(
            x_real_seq.T,
            pos_reference.T,
            errnorm_list,
            max_colorbar1,
            ax,
            title,
            start=0,
            end=-1,
            stp=2,
        )

    legend_elements = [
        Line2D([0], [0], linestyle="-.", color="k", lw=1, label="Reference"),
    ]

    norm_V = colors.Normalize(0.0, max_colorbar1)
    ax11.legend(handles=legend_elements, loc="best", framealpha=1)
    fig1.colorbar(
        cm.ScalarMappable(norm=norm_V, cmap=cm.turbo),
        cax=plt.axes([0.94, 0.35, 0.01, 0.3]),
        orientation="vertical",
        label="Tracking error [m]",
        fraction=0.15,
    )
    # plt.show()
    fig1.savefig("img/{}_tracking.png".format(CASE), dpi=300)

    """Plot State Trajectories"""
    fig2 = plt.figure(figsize=(12, 8))
    fig3 = plt.figure(figsize=(12, 8))
    fig4 = plt.figure(figsize=(12, 8))
    fig5 = plt.figure(figsize=(12, 8))
    fig6 = plt.figure(figsize=(12, 8))
    fig7 = plt.figure(figsize=(12, 8))
    fig8 = plt.figure(figsize=(12, 8))
    fig_list = [fig2, fig3, fig4, fig5, fig6, fig7, fig8]

    max_colorbar1 = max(max_ctrlerr_list[1:]) + 0.01  # exclude baseline
    for i, title, fig_ in zip(range(len(traj_dir_list)), title_list, fig_list):
        tracking = state_list[i]
        reference = xref_list[i]
        plot_states(tracking.T, reference.T, fig_, title)

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
