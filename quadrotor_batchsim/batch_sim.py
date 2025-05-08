import numpy as np
import argparse
import json
import os
from multiprocessing import Pool

"""
Env Module:
1. Vectorized Simulation Model
2. Disturbance Generator
"""


class VecEnv:
    def __init__(
        self,
        model_sim,
        model_nomi,
        batch_size,
        dt,
        x0_absbounds=[0.1] * 3 + [0.5] * 3 + [np.pi / 4] * 3 + [5.0] * 3 + [0.0] * 6,
    ):
        """
        Vectorized env for batch & parallel simulaion:
            1. domain randomization: random initial states around a reference trajectory (polynomial);
            2. batch_size = 1 refers to single model simulation;
            3. multiprocessing acceleration of batch simulation.
        """
        self.model_sim = model_sim
        self.model_nomi = model_nomi
        self.batch_size = batch_size
        self.h = dt
        # Rand bounds for domain randomization
        self.x0_absbounds = x0_absbounds

    def __create_init_states(self, x0_central, random_seed=None):
        x0_set = []
        if random_seed:
            np.random.seed(random_seed)  # set seed
        for idx in range(x0_central.shape[0]):
            samples = np.random.rand(self.batch_size) * 2 - 1
            samples_sized = self.x0_absbounds[idx] * samples
            x0_set += [samples_sized + x0_central[idx]]
        x0_set = np.vstack(x0_set).reshape((x0_central.shape[0], -1))

        return x0_set

    def single_rollout(self, x0, refk_seq, disturb_seq):
        xk = x0
        xk_nomi = self.model_sim.get_nomi_x(xk)
        uk = self.model_sim.get_nomi_u(xk, refk_seq[:, 0])
        xk_seq = [xk_nomi.reshape((-1, 1))]
        uk_seq = [uk.reshape((-1, 1))]
        # compute dx_real
        dx_real_seq = [
            self.model_sim.openloop_num(xk_nomi, uk, disturb_seq[:, 0]).reshape((-1, 1))
        ]
        # compute dx_nomi
        dx_nomi_seq = [self.model_nomi.openloop_num(xk_nomi, uk).reshape((-1, 1))]

        for ii in range(refk_seq.shape[1]):
            xk1 = self.model_sim.cldyn_num_exRK4(
                xk, refk_seq[:, ii], self.h, disturb_seq[:, ii]
            )
            xk_nomi = self.model_sim.get_nomi_x(xk)
            uk = self.model_sim.get_nomi_u(xk, refk_seq[:, ii])
            xk_seq += [xk_nomi.reshape((-1, 1))]
            uk_seq += [uk.reshape((-1, 1))]
            # compute dx_real
            dx_real_seq += [
                self.model_sim.openloop_num(xk_nomi, uk, disturb_seq[:, ii]).reshape(
                    (-1, 1)
                )
            ]
            # compute dx_nomi
            dx_nomi_seq += [self.model_nomi.openloop_num(xk_nomi, uk).reshape((-1, 1))]

            xk = xk1

        # shape: (x_dim, traj_len, batch_size)
        xk_seq = np.hstack(xk_seq)
        uk_seq = np.hstack(uk_seq)
        dx_real_seq = np.hstack(dx_real_seq)
        dx_nomi_seq = np.hstack(dx_nomi_seq)

        return xk_seq, uk_seq, dx_real_seq, dx_nomi_seq

    def concat_rollouts(self, x0_central, refk_seq, disturb_seq, state_rand_seed=None):
        # Simulate batch cases and compute dx
        x0_set = self.__create_init_states(x0_central, random_seed=state_rand_seed)
        xk_seq_batch = []
        dx_real_seq_batch = []
        dx_nomi_seq_batch = []

        for x0 in np.rollaxis(x0_set, 1):

            xk_seq, uk_seq, dx_real_seq, dx_nomi_seq = self.single_rollout(
                x0, refk_seq, disturb_seq
            )

            xk_seq_batch += [xk_seq]
            uk_seq_batch += [uk_seq]
            dx_real_seq_batch += [dx_real_seq]
            dx_nomi_seq_batch += [dx_nomi_seq]

        # temp save
        self.refk_seq = refk_seq
        self.disturb_seq = disturb_seq
        self.xk_seq_batch = np.dstack(xk_seq_batch)
        self.uk_seq_batch = np.dstack(uk_seq_batch)
        self.dx_real_seq_batch = np.dstack(dx_real_seq_batch)
        self.dx_nomi_seq_batch = np.dstack(dx_nomi_seq_batch)

        return (
            np.dstack(xk_seq_batch),
            np.hstack(uk_seq),
            np.dstack(dx_real_seq_batch),
            np.dstack(dx_nomi_seq_batch),
        )

    def concat_rollouts_multi(
        self,
        x0_central,
        refk_seq,
        disturb_seq,
        state_rand_seed=None,
        process_use=1,
    ):
        # Simulate batch cases and compute dx
        x0_set = self.__create_init_states(x0_central, random_seed=state_rand_seed)
        xk_seq_batch = []
        uk_seq_batch = []
        dx_real_seq_batch = []
        dx_nomi_seq_batch = []

        input_list = [(x0, refk_seq, disturb_seq) for x0 in np.rollaxis(x0_set, 1)]

        pool = Pool(processes=process_use)
        output_list = pool.starmap(self.single_rollout, input_list)

        self.refk_seq = refk_seq
        self.disturb_seq = disturb_seq
        xk_seq_batch, uk_seq_batch, dx_real_seq_batch, dx_nomi_seq_batch = zip(
            *output_list
        )
        self.xk_seq_batch = np.dstack(xk_seq_batch)
        self.uk_seq_batch = np.dstack(uk_seq_batch)
        self.dx_real_seq_batch = np.dstack(dx_real_seq_batch)
        self.dx_nomi_seq_batch = np.dstack(dx_nomi_seq_batch)
        # print(self.xk_seq_batch.shape, self.dx_real_seq_batch.shape, self.dx_nomi_seq_batch.shape)

        return (
            self.xk_seq_batch,
            self.uk_seq_batch,
            self.dx_real_seq_batch,
            self.dx_nomi_seq_batch,
        )

    def plot_batch(self, id_, dir_):
        # plot batch trajectories
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(11, 5))
        plt.subplots_adjust(0.05, 0.1, 0.9, 0.9, wspace=0.4)
        # plt.axis("off")
        ax_1 = fig.add_subplot(121, projection="3d")
        ax_1.view_init(elev=28, azim=-45)
        ax_1.invert_zaxis()
        # ax_1.set_xlabel("x [m]")
        # ax_1.set_ylabel("y [m]")
        # ax_1.set_zlabel("z [m]")
        plt.gca().set_box_aspect(
            (
                max(abs(self.refk_seq[0, :])),
                max(abs(self.refk_seq[1, :])),
                max(abs(self.refk_seq[2, :])),
            )
        )

        for xk_seq in np.rollaxis(self.xk_seq_batch, 2):
            ax_1.plot(
                self.refk_seq[0, :],
                self.refk_seq[1, :],
                self.refk_seq[2, :],
                c="k",
                linestyle="-.",
            )
            ax_1.plot(xk_seq[0, :], xk_seq[1, :], xk_seq[2, :], c="r", alpha=0.5)

        # plot disturbances
        # plt.axis("on")
        ax_2 = fig.add_subplot(122)
        ax_2.set_xlabel("N")
        ax_2.set_ylabel("Disturbance")
        for i in range(self.disturb_seq.shape[0]):
            ax_2.plot(self.disturb_seq[i, :], label="Output {}".format(i + 1))

        ax_2.legend()
        ax_2.grid()
        plt.savefig("{0}/batch_sim_{1}".format(dir_, id_), dpi=300)
        # plt.show()


class Disturbance_Generator:
    def __init__(
        self,
        out_dim,
        rand_seed_list=None,
        frequency_levels=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
        magnitude_bounds=[1.0] * 6,
    ):
        """Generate disturbance with random fourier series"""
        self.out_dim = out_dim
        # Disturbance with different frequency (Hz)
        self.frequency_levels = np.array(frequency_levels)
        # Magnitude Bound
        self.magnitude_bounds = np.array(magnitude_bounds)
        if len(self.frequency_levels) != len(self.magnitude_bounds):
            raise IndexError(
                "the number of frequencies should be equal to that of the magitudes!"
            )
        self.mag_randseeds = rand_seed_list
        self.__sample_magnitudes(rand_seed_list)

    def __sample_magnitudes(self, rand_seed_list):
        if rand_seed_list:
            if len(rand_seed_list) != self.out_dim:
                raise IndexError(
                    "the amount of random seeds should be equal out the output dim!"
                )
        magnitudes_set = []
        for i in range(self.out_dim):
            if rand_seed_list:
                np.random.seed(rand_seed_list[i])
            rand_scale = np.random.rand(len(self.frequency_levels)) * 2 - 1
            magnitudes = np.multiply(
                rand_scale,
                self.magnitude_bounds,
            )
            magnitudes_set += [magnitudes.reshape((-1, 1))]
        return np.hstack(magnitudes_set)

    def output_disturbance(self, t_seq):
        # sample magnitude
        magnitudes_set = self.__sample_magnitudes(self.mag_randseeds)
        # go through all output channels
        disturb_seq = []
        for ii in range(self.out_dim):
            mags = magnitudes_set[:, ii]
            print(mags)
            disturb = 0.0
            for freq, mag in zip(self.frequency_levels, mags):
                if freq == 0.0:
                    disturb += mag * np.ones_like(t_seq)
                else:
                    disturb += mag * np.sin(freq * 2 * np.pi * t_seq)
            disturb_seq += [disturb]

        disturb_seq = np.vstack(disturb_seq)  # roll: disturbance seq of one output
        return disturb_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setups_path",
        type=str,
        default="metatest_test_setups.json",
        help="json file path for setups",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="metatest_test",
        help="save name for simulated data",
    )
    parser.add_argument(
        "--process_use", type=int, default=1, help="cpu core use for simulation"
    )
    parser.add_argument(
        "--vis", type=bool, default=True, help="whether to plot simulation result"
    )

    opt = parser.parse_args()

    setups = {}
    with open(opt.setups_path, "r", encoding="utf-8") as f:
        setups = json.load(f)

    traj_num = setups["traj_num"]
    batch_num = setups["batch_num"]
    timestep = setups["time_step"]
    force_output_dim = setups["force_output_dim"]
    force_frequency_levels = setups["force_frequency_levels"]
    force_magnitude_bounds = setups["force_magnitude_bounds"]
    torque_output_dim = setups["torque_output_dim"]
    torque_frequency_levels = setups["torque_frequency_levels"]
    torque_magnitude_bounds = setups["torque_magnitude_bounds"]
    max_waypoint_num = setups["traj_max_waypoints"]
    waypoint_xbound = setups["waypoint_xbound"]
    waypoint_ybound = setups["waypoint_ybound"]
    waypoint_zbound = setups["waypoint_zbound"]
    dis_bound = setups["discretization_bound"]
    max_refu_mag = setups["max_refu_mag"]
    u_bound = setups["u_bound"]

    from models.model_nomi import Quadrotor
    from models.model_sim import Quadrotor_Sim
    from aux_module.trajectory_utils import *

    model_nomi = Quadrotor()
    model_sim = Quadrotor_Sim()
    vec_env = VecEnv(
        model_sim=model_sim, model_nomi=model_nomi, batch_size=batch_num, dt=timestep
    )
    force_generator = Disturbance_Generator(
        out_dim=force_output_dim,
        frequency_levels=force_frequency_levels,
        magnitude_bounds=force_magnitude_bounds
    )
    # torque_generator = Disturbance_Generator(
    #     out_dim=torque_output_dim,
    #     frequency_levels=torque_frequency_levels,
    #     magnitude_bounds=torque_magnitude_bounds,
    # )
    trajopt = Polynomial_TrajGen(model=model_nomi)

    traj_id = 1
    while traj_id <= traj_num:
        # Random Trajectory Optimization
        pos_waypoints = trajopt.set_rand_waypoints(
            max_waypoint_num, waypoint_xbound, waypoint_ybound, waypoint_zbound
        )
        # Generate random command trajectories
        t_init = 0.0
        refx0 = np.hstack([pos_waypoints[0, :], np.array([0] * 9)])
        refxf = np.hstack([pos_waypoints[-1, :], np.array([0] * 9)])
        trajopt.set_refxBoundCond(refx0, refxf)

        N = trajopt.set_rand_discreteN(
            h=timestep, N_min=dis_bound[0], N_max=dis_bound[1]
        )
        trajopt.set_rand_refuBoxCons(max_refu_mag)
        trajopt.set_rand_modeluBoxCons(min_u=u_bound[0], max_u=u_bound[1])

        trajopt.NLP_Prepare()

        sol, if_success = trajopt.NLP_FormAndSolve(Eq_Relax=0.0)

        if if_success:
            refx_opt = trajopt.get_refxopt(sol)
            refx = ca.vertcat(
                refx0.reshape((-1, refx0.shape[0])),
                refx_opt[:-1],
            )
            refk_seq = refx.toarray().T

            x0 = np.hstack([pos_waypoints[0, :], [0] * 15])

            # Construct Disturbance
            t_seq = np.array([timestep * t for t in range(N)])
            force_seq = force_generator.output_disturbance(t_seq)
            # torque_seq = torque_generator.output_disturbance(t_seq)
            torque_seq = np.zeros_like(force_seq)
            disturb_seq = np.vstack([force_seq, torque_seq])

            # Batch Rollout
            xk_seq_batch, uk_seq_batch, dx_real_seq_batch, dx_nomi_seq_batch = (
                vec_env.concat_rollouts_multi(x0, refk_seq, disturb_seq, process_use=opt.process_use)
            )

            # Plot Current Batch
            if opt.vis:
                save_dir = "./img/{}".format(opt.save_name)
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                vec_env.plot_batch(id_=traj_id, dir_=save_dir)

            # Save Data
            array_dict = {
                "x_seq_batch": xk_seq_batch,
                "u_seq_batch": uk_seq_batch,
                "dx_real_seq_batch": dx_real_seq_batch,
                "dx_nomi_seq_batch": dx_nomi_seq_batch,
                "ref_seq_batch": refk_seq,
            }

            save_dir = "./data/{}".format(opt.save_name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            np.savez(save_dir + "/batch_sim_{}.npz".format(traj_id), **array_dict)

            traj_id += 1
