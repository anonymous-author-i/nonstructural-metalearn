import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_data(dir_path, state_select, data="learn"):
    disturb_batches = []
    for file in os.listdir(dir_path):
        if file.endswith(".npz") and data=="learn":  # batch dataset for learning
            # batch: (state_dim, traj_len, batch_size)
            file_path = os.path.join(dir_path, file)
            data = np.load(file_path)
            dx_real_seq_batch = data["dx_real_seq_batch"]
            dx_nomi_seq_batch = data["dx_nomi_seq_batch"]
            disturb_seq_batch = (
                dx_real_seq_batch[state_select] - dx_nomi_seq_batch[state_select]
            )
            disturb_batches += [disturb_seq_batch]
        elif file.endswith(".npz") and data=="sim": # estimation and control simulation
            # sequence: (state_dim, traj_len)
            file_path = os.path.join(dir_path, file)
            data = np.load(file_path)
            dx_real_seq = data["dx_real"]
            dx_nomi_seq = data["dx_nomi"]
            disturb_seq = dx_real_seq[state_select] - dx_nomi_seq[state_select]
            disturb_seq_batch = np.expand_dims(
                disturb_seq, axis=-1
            )  # add batch dim = 1
            disturb_batches += [disturb_seq_batch]

    return disturb_batches


def get_norm_and_rate(batch_list, dt):
    batch_flatten = []
    rate_batch_flatten = []
    norm_batch_flatten = []

    for batch in batch_list:
        # batch: (state_dim, traj_len, batch_size)
        rate_batch = []
        rate_batch += [np.zeros_like(batch[:, 0, :])]
        for i in range(batch.shape[1] - 1):
            rate_batch += [(batch[:, i + 1, :] - batch[:, i, :]) / dt]

        batch_flatten += [batch.reshape((batch.shape[0], -1))]
        norm_batch_flatten += [np.linalg.norm(batch, axis=0).reshape((1, -1))]
        rate_batch_flatten += [np.hstack(rate_batch)]

    return (
        np.hstack(batch_flatten),
        np.hstack(norm_batch_flatten),
        np.hstack(rate_batch_flatten),
    )


if __name__ == "__main__":
    # Load and process
    disturb_learn = load_data("sim_data/metalearn", state_select=[3, 4, 5], data='learn')
    disturb_sim = load_data("sim_lemniscate_esti", state_select=[3, 4, 5], data='sim')
    vec_flatten_learn, norm_flatten_learn, rate_flatten_learn = get_norm_and_rate(
        batch_list=disturb_learn, dt=0.02
    )
    len_learn = vec_flatten_learn.shape[1]
    vec_flatten_sim, norm_flatten_sim, rate_flatten_sim = get_norm_and_rate(
        batch_list=disturb_sim, dt=0.02
    )
    len_sim = vec_flatten_sim.shape[1]

    # make stat_dict
    stat_dict = {
        "Magnitude": list(vec_flatten_learn[0, :])
        + list(vec_flatten_learn[1, :])
        + list(vec_flatten_learn[2, :])
        + list(vec_flatten_sim[0, :])
        + list(vec_flatten_sim[1, :])
        + list(vec_flatten_sim[2, :]),
        "Rate": list(rate_flatten_learn[0, :])
        + list(rate_flatten_learn[1, :])
        + list(rate_flatten_learn[2, :])
        + list(rate_flatten_sim[0, :])
        + list(rate_flatten_sim[1, :])
        + list(rate_flatten_sim[2, :]),
        "Dataset": ["Learning"] * len_learn * 3 + ["Simulation"] * len_sim * 3,
        "Direction": ["X"] * len_learn
        + ["Y"] * len_learn
        + ["Z"] * len_learn
        + ["X"] * len_sim
        + ["Y"] * len_sim
        + ["Z"] * len_sim,
    }
    df = pd.DataFrame.from_dict(stat_dict)

    # cut-off
    filter = abs(df["Rate"]) < 15
    df.where(filter, inplace=True)

    # plot
    sns.set_theme("paper", "darkgrid", font="Calibri", font_scale=1.5)
    g1 = sns.FacetGrid(data=df, col="Direction", hue="Dataset", height=2.5)
    g1.map(
        sns.histplot,
        "Magnitude",
        stat="density",
        binwidth=0.15,
        alpha=0.6,
        element="poly",
    )
    g1.savefig("img/magnitude_gap_dist", dpi=300)
    g2 = sns.FacetGrid(data=df, col="Direction", hue="Dataset", height=2.5)
    g2.map(
        sns.histplot, "Rate", stat="density", binwidth=0.15, alpha=0.6, element="poly"
    )
    g2.axes_dict['Z'].legend()
    g2.savefig("img/rate_gap_dist", dpi=300)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # sns.violinplot(data=df, x="Direction", y="Magnitude", hue="Dataset", ax=ax1)
    # sns.violinplot(data=df, x="Direction", y="Rate", hue="Dataset", ax=ax2)
    # plt.show()
