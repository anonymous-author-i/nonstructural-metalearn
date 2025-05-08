import numpy as np
from dynamics.model_nomi import Quadrotor


class Quadrotor_DFBC(Quadrotor):
    def __init__(self):
        super(Quadrotor_DFBC, self).__init__()
        self.__control_param()
        self.pos_out_dim = 3
        self.ctrl_slackvar_dim = 6

    def __control_param(self):
        self.pos_gain = np.diag(np.array([1.0, 1.0, 0.7])) * 2
        self.vel_gain = self.pos_gain * 4
        self.eul_gain = np.diag(np.array([10.0, 10.0, 4.0]))
        self.omega_P = np.diag(np.array([40.0, 40.0, 16.0]))
        self.omega_I = np.diag(np.array([10.0, 10.0, 5.0]))
        self.omega_D = np.diag(np.array([0.5, 0.5, 0.0]))

    def __rpy2DCMbe_num(self, eul_rpy):
        # eul_zyx <- eul_rpy 1,3 switch
        eul_z = eul_rpy[2]
        eul_y = eul_rpy[1]
        eul_x = eul_rpy[0]
        R1 = np.vstack(
            [
                np.hstack([np.cos(eul_z), -np.sin(eul_z), 0]),
                np.hstack([np.sin(eul_z), np.cos(eul_z), 0]),
                np.hstack([0, 0, 1]),
            ]
        )
        R2 = np.vstack(
            [
                np.hstack([np.cos(eul_y), 0, np.sin(eul_y)]),
                np.hstack([0, 1, 0]),
                np.hstack([-np.sin(eul_y), 0, np.cos(eul_y)]),
            ]
        )
        R3 = np.vstack(
            [
                np.hstack([1, 0, 0]),
                np.hstack([0, np.cos(eul_x), -np.sin(eul_x)]),
                np.hstack([0, np.sin(eul_x), np.cos(eul_x)]),
            ]
        )
        return R1 @ R2 @ R3

    def __dEul2omega_num(self, dEul_des, Eul):
        # Strap Down Equations
        domega_xdes = dEul_des[0] - (np.sin(Eul[1]) * dEul_des[2])
        domega_ydes = (dEul_des[1] * np.cos(Eul[0])) + (
            dEul_des[2] * np.sin(Eul[0]) * np.cos(Eul[1])
        )
        domega_zdes = -(dEul_des[1] * np.sin(Eul[0])) + (
            dEul_des[2] * np.cos(Eul[0]) * np.cos(Eul[1])
        )
        return np.hstack([domega_xdes, domega_ydes, domega_zdes])

    def __invert_eul_num(self, moment_des, omega):
        m1 = moment_des[0] + omega[1] * omega[2] * (self.Izz - self.Iyy)
        m2 = moment_des[1] + omega[0] * omega[2] * (self.Ixx - self.Izz)
        m3 = moment_des[2] + omega[0] * omega[1] * (self.Iyy - self.Ixx)
        return np.hstack([m1, m2, m3])

    def __derative3(self, interstate, x):
        # c=0.05 tf = s/(c*s+1)
        d_interstate = -10 * np.eye(3) @ interstate + 8 * np.eye(3) @ x
        x_der = -12.5 * np.eye(3) @ interstate + 10 * np.eye(3) @ x
        return d_interstate, x_der

    def ctrlmap_num(self, x, ref, aux, d_esti=np.zeros(3)):
        # demux
        pos = x[0:3]
        vel = x[3:6]
        eul = x[6:9]
        omega = x[9:12]
        pos_ref = ref[0:3]
        vel_ref = ref[3:6]
        acc_ref = ref[6:9]
        jer_ref = ref[9:12]

        omega_err_inte, omega_err_ds = aux[0:3], aux[3:6]

        # Translational Loop
        a_des = (
            acc_ref + self.pos_gain @ (pos_ref - pos) + self.vel_gain @ (vel_ref - vel)
        ) - d_esti
        # Obtain Desire Rebs
        Zb = np.hstack([-a_des[0], -a_des[1], 9.8 - a_des[2]]) / np.linalg.norm(
            np.hstack([-a_des[0], -a_des[1], 9.8 - a_des[2]])
        )

        Xc = np.hstack([np.cos(0.0), np.sin(0.0), 0.0])
        Yb_ = np.cross(Zb, Xc)
        Yb = Yb_ / np.linalg.norm(Yb_)
        Xb = np.cross(Yb, Zb)
        Reb_des = np.vstack([Xb, Yb, Zb]).T
        # Reb_des = ca.horzcat(Xb, Yb, Zb)

        # Obtain Desire Eul, Omega and dOmega
        # eul_des = 'ZYX' + 1,3 Switch
        eul_des = np.hstack(
            [
                np.arctan2(Reb_des[2, 1], Reb_des[2, 2]),
                np.arctan2(
                    -Reb_des[2, 0], np.sqrt(Reb_des[1, 0] ** 2 + Reb_des[0, 0] ** 2)
                ),
                np.arctan2(Reb_des[1, 0], Reb_des[0, 0]),
            ]
        )

        T = -self.m * np.dot(Zb, (a_des - np.hstack([0.0, 0.0, 9.8])))
        h1 = -self.m / T * (jer_ref - np.dot(Zb, jer_ref) * Zb)

        omega_des = np.hstack([-np.dot(h1, Yb), np.dot(h1, Xb), 0.0])

        # h2 = (
        #     -np.cross(omega_des, np.cross(omega_des, Zb))
        #     + self.m / T * np.dot(jer_ref, Zb) * np.cross(omega_des, Zb)
        #     + 2 * self.m / T * np.dot(Zb, jer_ref) * np.cross(omega_des, Zb)
        # )
        # domega_des = np.array(
        #     [-np.dot(h2, Yb), np.dot(h2, Xb), 0.0]
        # )  # yaw, dyaw, ddyaw = 0

        # Attitude Loop
        dEul_des = self.eul_gain @ (eul_des - eul)
        omega_err = omega_des - omega + self.__dEul2omega_num(dEul_des, eul)
        d_omega_err_ds, omega_err_der = self.__derative3(omega_err_ds, omega_err)
        att_out = (
            self.omega_P @ omega_err
            + self.omega_I @ omega_err_inte
            + self.omega_D @ omega_err_der
            + np.cross(omega, self.J @ omega)
            # + self.J @ domega_des
        )
        moment_des = self.J @ att_out
        tau = self.__invert_eul_num(moment_des, omega)

        MF = self.CoeffM @ np.hstack([T, tau[0], tau[1], tau[2]])
        # # Saturation
        # for i in range(MF.shape[0]):
        #     if self.u_lb[i] > MF[i]:
        #         MF[i] = self.u_lb[i]
        #     elif self.u_ub[i] < MF[i]:
        #         MF[i] = self.u_ub[i]
        #     else:
        #         pass

        daux = np.hstack([omega_err, d_omega_err_ds])
        return MF, daux, a_des, eul_des, omega_des


# class Quadrotor_ForceEstimator(Quadrotor):
#     def __init__(self, predictor_model, ff_enable, feedback_gain):
#         super(Quadrotor_ForceEstimator, self).__init__()
#         self.feedback_gain = feedback_gain
#         self.predictor = predictor_model
#         self.ff_enable = ff_enable
#         self.esti_slackvar_dim = 3

#     def force_esti_x(self, z_hat, x, a_des, ff_inputs, Theta):
#         # forward dynamics (vel-acc)
#         dv = a_des
#         # compute
#         if self.ff_enable: pred = self.predictor.predict(ff_inputs, Theta)
#         else: pred = np.zeros(self.esti_slackvar_dim)
#         dz_hat = -self.feedback_gain * (z_hat + pred + self.feedback_gain * x[3:6] + dv)
#         d_hat = z_hat + pred + self.feedback_gain * x[3:6]
#         return dz_hat, d_hat

#     def force_esti_dx(self, z_hat, disturb, ff_inputs, Theta):
#         if self.ff_enable: pred = self.predictor.predict(ff_inputs, Theta)
#         else: pred = np.zeros(3)
#         dz_hat = self.feedback_gain * (disturb - z_hat - pred)
#         d_hat = z_hat + pred
#         return dz_hat, d_hat