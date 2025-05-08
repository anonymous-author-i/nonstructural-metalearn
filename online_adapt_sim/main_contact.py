import numpy as np
import argparse
import json
import os
from sklearn.metrics import root_mean_squared_error as RMSE
from dynamics.model_nomi import Quadrotor
from dynamics.model_sim import Quadrotor_Sim
from model_wrapper import *
from estimator.meta_adaptive import *
from estimator.disturbance_observer import *
from estimator.l1_adaptive import *
from controller.dfbc import *
from utils import *

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
        default="meta_mlpt_h0_bl10",
        help="name of the learned encoder model",
    )
    parser.add_argument(
        "--discrete_h",
        type=float,
        default=0.02,
        help="discrete time step for simulation",
    )
    parser.add_argument(
        "--traj_speedrate",
        type=float,
        default=0.8,
        help="the aggressiveness of reference trajectory",
    )
    parser.add_argument("--device", default="cpu", help="device used for training")

    opt = parser.parse_args()
    # SIMSTEPS = opt.sim_timesteps
    TIMESTEP = opt.discrete_h
    TRAJ_SPEEDRATE = opt.traj_speedrate

    """ Simulation Setup """
    # Load setups
    setups = {}
    with open(
        "learned_models/" + opt.model_name + "/model_setup.json", "r", encoding="utf-8"
    ) as f:
        setups = json.load(f)
    settings = setups[opt.encoder]

    # Initialize nominal dynamics
    nomi_dyn = Quadrotor()

    # Make meta estimators
    adaptive_model = get_predictor_model(opt, setups)
    pth_path = "learned_models/" + opt.model_name + "/model_param.pth"
    adaptive_model.load_state_dict(torch.load(pth_path))
    adaptive_model.eval()
    predictor_adapt = PredictorShell_Adapt(
        settings, torch_model=adaptive_model, mode="normal"
    )
    adaptive_model = Model_Wrapper(
        nominal_model=nomi_dyn, predictor_model=predictor_adapt, discrete_h=TIMESTEP
    )
    MDE_full = MetaAdaptiveEstimator(
        ff_model=adaptive_model,
        ff_enable=True,
        adapt_gain=2.0
        * np.eye(len(settings["output_select"]) * settings["hidden_sizes"][-1]),
        feedback_gain=2.0,
        concurrent_learn_len=5,
    )
    MDE_ff = MetaAdaptiveEstimator(
        ff_model=adaptive_model,
        ff_enable=True,
        adapt_gain=2.0
        * np.eye(len(settings["output_select"]) * settings["hidden_sizes"][-1]),
        feedback_gain=0.0,
        concurrent_learn_len=5,
    )

    # Make disturbance observers
    flat_model = Model_Wrapper(
        nominal_model=nomi_dyn, predictor_model=None, discrete_h=TIMESTEP
    )
    # Load disturbance
    disturbance_dict = np.load("contact_force.npz")
    disturb_seq = disturbance_dict["case1"]
    SIMSTEPS = disturb_seq.shape[1]

    Baseline = DisturbanceObserver(
        ff_model=flat_model,
        output_select=settings["output_select"],
        feedback_gain=2.0 * np.eye(len(settings["output_select"])),
        dt=TIMESTEP,
    )
    DO = DisturbanceObserver(
        ff_model=flat_model,
        output_select=settings["output_select"],
        feedback_gain=2.0 * np.eye(len(settings["output_select"])),
        dt=TIMESTEP,
    )

    # Make l1 adaptive estimator
    L1Adapt = L1AdaptiveEstimator(
        nominal_model=nomi_dyn,
        output_select=settings["output_select"],
        adapt_gain=4.0 * np.eye(len(settings["output_select"])),
        filter_gain=0.25 * np.eye(len(settings["output_select"])),
        dt=TIMESTEP,
    )

    # Prepare simulation environments
    sim_dyn = Quadrotor_Sim()
    # sim_dyn.aero_D = np.diag([0.6, 0.6, 0.1])
    # sim_dyn.m_actual += 0.3
    # sim_dyn.Ixx_actual += 1e-3
    # sim_dyn.Iyy_actual += 1e-3

    # Trajectory info
    def get_reference(t):
        pvaj = np.array([0] * 2 + [-1.5] + [0] * 9)
        return pvaj

    # Get position reference trajectory for plotting
    pv_reference = []
    for i in range(SIMSTEPS):
        pvaj = get_reference(i * TIMESTEP)
        pv_reference += [pvaj[:6].reshape((-1, 1))]
    pv_reference = np.hstack(pv_reference)

    """Simulation Setup"""
    # Prepare controllers and closed-loop model
    dfbc_base = Quadrotor_DFBC()
    controller_baseline = ClosedLoop_Wrapper(
        openloop_dynamics=sim_dyn,
        controller=dfbc_base,
        discrete_h=TIMESTEP,
        ffcomp_on=False,
    )
    closed_loop_sys = ClosedLoop_Wrapper(
        openloop_dynamics=sim_dyn,
        controller=dfbc_base,
        discrete_h=TIMESTEP,
        ffcomp_on=True,
    )
    esti_cases = [
        (0, DO, "DO"),
        (1, L1Adapt, "L1-Adapt"),
        (2, MDE_ff, "Meta-Adapt"),
        (3, MDE_full, "Meta-Adapt-FC"),
        (4, MDE_full, "Meta-LS-FC"),
    ]

    # Simulation Main Function
    def sim_main(idx, closed_loop_sys, estimator, disturb_seq, verbose=False):
        # Get initial values
        x0 = nomi_dyn.ref2x_map(get_reference(0)).full()
        concat_state = np.vstack([x0, np.zeros((closed_loop_sys.ctrl_slackvar_dim, 1))])
        concat_state = np.squeeze(concat_state, axis=1)
        z_hat = np.zeros(len(settings["output_select"]))
        d_hat = np.zeros(len(settings["output_select"]))

        # Add lists
        x_real = []
        dx_real = []
        dx_pred = []
        dx_nomi = []
        eul_des = []
        omega_des = []
        t = 0
        xk = np.squeeze(x0, axis=1)
        uk = nomi_dyn.m * 9.8 / 4 * np.ones(nomi_dyn.nomi_u_dim)
        x_real += [x0]
        track_err = np.zeros(3)

        # initialize the online learner for meta-adaptive methods
        if idx >= 2:
            Theta_adapt = np.zeros(
                len(settings["output_select"]) * settings["hidden_sizes"][-1]
            )
            ff_input_init = np.hstack(
                [
                    xk[adaptive_model.predictor.state_select],
                    uk[adaptive_model.predictor.control_select],
                ]
            ).reshape((-1, 1))
            estimator.init_learner_and_stack(
                init_ffinput=ff_input_init,
                init_dx=sim_dyn.openloop_num(xk, uk).reshape((-1, 1)),
            )

        for i in range(SIMSTEPS):  # main loop for simulation
            if i % 10 == 0 and verbose:
                print("Current Case: {0}, step {1}/{2}".format(idx, i, SIMSTEPS))

            # Inject disturbance
            sim_dyn.disturb = np.hstack([disturb_seq[:, i], [0] * 3])
            # closed_loop_sys.sim_model.disturb = np.hstack([disturb_seq[:, i], [0] * 3])
            # Get dx
            dx_real_i = sim_dyn.openloop_num(xk, uk)
            dx_nomi_i = nomi_dyn.openloop_num(xk, uk)
            dx_pred_i = dx_nomi_i.copy()
            dx_pred_i[settings["output_select"]] = (
                dx_nomi_i[settings["output_select"]] + d_hat
            )
            dx_pred_i = dx_pred_i.reshape((-1, 1))

            # Online estimation
            if 0 <= idx <= 1:
                z_hat1, d_hat = estimator.d_esti_x(z_hat, xk, uk)
            elif idx >= 2:
                a_des = closed_loop_sys.a_des
                disturb_i = dx_real_i[settings["output_select"]] - a_des
                disturb_i = disturb_i.reshape((-1, 1))
                if idx == 4:
                    Theta_adapt1 = estimator.direct_least_square(reg=0.01)
                else:
                    Theta_adapt1 = estimator.concurrent_learn(Theta_adapt)
                z_hat1, d_hat = estimator.d_esti_dx(
                    z_hat, dx_real_i, xk, uk, Theta_adapt1
                )

                # Get predictor inputs, real dx
                ff_input_i = np.hstack(
                    [
                        xk[adaptive_model.predictor.state_select],
                        uk[adaptive_model.predictor.control_select],
                    ]
                ).reshape((-1, 1))
                # use real dx feedback
                estimator.store_data(ff_input_i, dx_real_i.reshape((-1, 1)), disturb_i)

            # Rollout out simulation model (closed-loop)
            refk = get_reference(i * TIMESTEP)
            track_err = refk[:3] - xk[:3]
            concat_state1 = closed_loop_sys.closedloop_num_exRK4(
                concat_state, refk, d_hat
            )
            uk = closed_loop_sys.u
            eul_des += [closed_loop_sys.eul_des.reshape((-1, 1))]
            omega_des += [closed_loop_sys.omega_des.reshape((-1, 1))]
            xk1 = concat_state1[0 : closed_loop_sys.x_dim]

            # Store sim state
            dx_real += [dx_real_i.reshape((-1, 1))]
            dx_pred += [dx_pred_i.reshape((-1, 1))]
            dx_nomi += [dx_nomi_i.reshape((-1, 1))]
            x_real += [xk1.reshape((-1, 1))]

            # Update sim state
            concat_state = concat_state1
            z_hat = z_hat1
            xk = xk1
            # Update time step
            t += TIMESTEP

            # Update Theta
            if idx >= 2:
                Theta_adapt = Theta_adapt1

        return (
            np.hstack(x_real),
            np.hstack(eul_des),
            np.hstack(omega_des),
            np.hstack(dx_real),
            np.hstack(dx_pred),
            np.hstack(dx_nomi),
        )

    """Go Estimation Simulation"""
    for idx, estimator, name in esti_cases:
        # Simulate
        x_real, eul_des, omega_des, dx_real, dx_pred, dx_nomi = sim_main(
            idx, controller_baseline, estimator, disturb_seq
        )

        # Save Trajectory
        array_dict = {
            "x_real": x_real[:, 50:-50],
            "x_ref": np.vstack([pv_reference, eul_des, omega_des])[:, 50:-50],
            "pos_reference": pv_reference[0:3, 50:-50],
            "dx_real": dx_real[:, 50:-50],
            "dx_pred": dx_pred[:, 50:-50],
            "dx_nomi": dx_nomi[:, 50:-50],
        }
        np.savez("sim_contact_esti/{0}_{1}.npz".format(idx, name), **array_dict)

        start = 20

        # visualize
        fig1 = plt.figure()
        dx_pred = dx_pred.T
        dx_real = dx_real.T
        dx_nomi = dx_nomi.T
        disturb_pred = dx_pred - dx_nomi
        # plt.plot(disturb_pred[:, [3, 4, 5]], label="pred")
        # plt.plot(disturb_seq.T / sim_dyn.m, label="real", linestyle="--")
        # plt.grid()
        # plt.legend()
        # plt.show()

        print(
            "Esti Estimation RMSE:",
            RMSE(disturb_pred[:, [3, 4, 5]], disturb_seq.T/ sim_dyn.m),
            # RMSE(dx_pred[start:, [3, 4, 5]], dx_real[start:, [3, 4, 5]]),
        )

    # '''Go Controller Simulation'''
    # for idx, estimator, name in esti_cases:
    #     # Simulate
    #     x_real, eul_des, omega_des, dx_real, dx_pred, dx_nomi = sim_main(
    #         idx, controller_comp, estimator, disturb_seq
    #     )

    #     # Save Trajectory
    #     array_dict = {
    #             "x_real": x_real,
    #             "x_ref": np.vstack([pv_reference, eul_des, omega_des]),
    #             "pos_reference": pv_reference[0:3, :],
    #             "dx_real": dx_real,
    #             "dx_pred": dx_pred,
    #             "dx_nomi": dx_nomi,
    #         }
    #     np.savez("sim_contact_ctrl/{0}_{1}.npz".format(idx, name), **array_dict)

    #     start = 20
    #     print("Ctrl Estimation RMSE:", RMSE(dx_pred[[3, 4, 5], start:], dx_real[[3, 4, 5], start:]))

    #     # visualize
    #     fig1 = plt.figure()
    #     dx_pred = dx_pred.T
    #     dx_real = dx_real.T
    #     dx_nomi = dx_nomi.T
    #     plt.plot(dx_pred[:, [3, 4, 5]], label="pred")
    #     plt.plot(dx_real[:, [3, 4, 5]], label="real", linestyle="--")
    #     plt.grid()
    #     plt.legend()
    #     plt.show()
