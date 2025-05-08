import numpy as np
import casadi as ca


class Quadrotor_NODE:
    def __init__(self, params):
        '''
        Open-loop + Closed-loop model of a quadrotor:

        The model includes:
            1. Inertia and Mass Bias
            2. Rotor Drag
            3. Others if needed

        - Open-loop Dynamics:
            dp = v
            dv = a + AeroDrag
            dEul = W(Eul) @ pqr
            dpqr = J^-1 (J pqr x pqr + tau)

            x = [p, v, eul, pqr]
            u = [T1, T2, T3, T4]
            dx = f(x, u)
        '''
        self.__dynamic_param()
        self.__control_param()
        self.__saturation_params()

        # Model Information
        self.nomi_x_dim = 12
        self.nomi_u_dim = 4
        self.x_dim = self.nomi_x_dim
        self.u_dim = self.nomi_u_dim
        
        # NN ode setting
        self.input_size = 6
        self.hidden_size = 36
        self.output_size = 3

        # Initialize and Compute param dimension
        x_test = ca.DM.ones(6)
        self.NN_model(x_test, params=ca.DM.ones(10000))  # sufficient param dim
        print("param size: ", self.params_dim)

        self.params = params

    def load_params_for_OC(self, params):
        # This is for the usage of model
        self.params = params

    def __dynamic_param(self):
        self.m = 0.83
        self.Ixx = 3e-3
        self.Iyy = 3e-3
        self.Izz = 4e-3
        self.__compute_J()

        torque_coef = 0.01
        arm_length = 0.150
        self.CoeffM = np.array(
            [
                [
                    0.25,
                    -0.353553 / arm_length,
                    0.353553 / arm_length,
                    0.25 / torque_coef,
                ],
                [
                    0.25,
                    0.353553 / arm_length,
                    -0.353553 / arm_length,
                    0.25 / torque_coef,
                ],
                [
                    0.25,
                    0.353553 / arm_length,
                    0.353553 / arm_length,
                    -0.25 / torque_coef,
                ],
                [
                    0.25,
                    -0.353553 / arm_length,
                    -0.353553 / arm_length,
                    -0.25 / torque_coef,
                ],
            ]
        )  # Ttau -> MF
        self.CoeffM_inv = np.linalg.inv(self.CoeffM)  # Ttau -> MF

    def __compute_J(self):
        self.J = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))
        self.J_inv = np.diag(np.array([1 / self.Ixx, 1 / self.Iyy, 1 / self.Izz]))

    def __control_param(self):
        self.pos_gain = np.diag(np.array([1.0, 1.0, 0.7])) * 2
        self.vel_gain = self.pos_gain * 4
        self.eul_gain = np.diag(np.array([10.0, 10.0, 4.0]))
        self.omega_P = np.diag(np.array([40.0, 40.0, 16.0]))
        self.omega_I = np.diag(np.array([10.0, 10.0, 5.0])) 
        self.omgea_D = np.diag(np.array([0.5, 0.5, 0.0]))

    def __saturation_params(self):
        self.u_lb = np.array([0.0, 0.0, 0.0, 0.0])
        self.u_ub = np.array([4.0, 4.0, 4.0, 4.0]) * 1.5

    def __linear_layer(self, x, param, dim_in, dim_out):
        W = param[0 : dim_in * dim_out].reshape((dim_out, dim_in))
        b = param[dim_in * dim_out : dim_in * dim_out + dim_out]
        return W @ x + b

    def __linear_layer_nobias(self, x, param, dim_in, dim_out):
        W = param[0 : dim_in * dim_out].reshape((dim_out, dim_in))
        return W @ x

    def NN_model(self, x, params):
        '''
        NN ode for learing the translational dynamics of a quadrotor
        '''
        x_ = x.reshape((-1,1))
        param_idx = 0
        param_idx_1 = 0
        # Input Layer
        dim_in = self.input_size
        dim_out = self.hidden_size
        param_idx_1 += dim_in * dim_out + dim_out
        param_ = params[param_idx:param_idx_1]
        param_idx = param_idx_1

        x_ = self.__linear_layer(x_, param_, dim_in, dim_out)
        x_ = np.tanh(x_)

        # Hidden Layer 1
        dim_in = self.hidden_size
        dim_out = self.hidden_size
        param_idx_1 += dim_in * dim_out + dim_out
        param_ = params[param_idx:param_idx_1]
        param_idx = param_idx_1

        x_ = self.__linear_layer(x_, param_, dim_in, dim_out)
        x_ = np.tanh(x_)

        # Hidden Layer 2
        dim_in = self.hidden_size
        dim_out = self.hidden_size
        param_idx_1 += dim_in * dim_out + dim_out
        param_ = params[param_idx:param_idx_1]
        param_idx = param_idx_1

        x_ = self.__linear_layer(x_, param_, dim_in, dim_out)
        x_ = np.tanh(x_)
        
        # Full Connected Layer
        dim_in = self.hidden_size
        dim_out = self.output_size
        # param_idx_1 += dim_in * dim_out + dim_out
        param_idx_1 += dim_in * dim_out
        param_ = params[param_idx:param_idx_1]
        param_idx = param_idx_1

        x_ = self.__linear_layer_nobias(x_, param_, dim_in, dim_out)

        self.params_dim = param_idx_1
        return x_
    
    def openloop_sym(self, x, MF):
        Ttau = self.CoeffM_inv @ MF

        dp = x[3:6]

        dvx = (
            -Ttau[0]
            / self.m
            * (ca.cos(x[8]) * ca.sin(x[7]) * ca.cos(x[6]) + ca.sin(x[8]) * ca.sin(x[6]))
        )
        dvy = (
            -Ttau[0]
            / self.m
            * (ca.sin(x[8]) * ca.sin(x[7]) * ca.cos(x[6]) - ca.cos(x[8]) * ca.sin(x[6]))
        )
        dvz = 9.8 - Ttau[0] / self.m * (ca.cos(x[6]) * ca.cos(x[7]))
        dv = ca.vertcat(dvx, dvy, dvz) - self.NN_model(x[3:9], self.params)

        deul = (
            ca.vertcat(
                ca.horzcat(1, ca.tan(x[7]) * ca.sin(x[6]), ca.tan(x[7]) * ca.cos(x[6])),
                ca.horzcat(0, ca.cos(x[6]), -ca.sin(x[6])),
                ca.horzcat(0, ca.sin(x[6]) / ca.cos(x[7]), ca.cos(x[6]) / ca.cos(x[7])),
            )
            @ x[9:12]
        )

        domega = self.J_inv @ (
            -ca.cross(x[9:12], self.J @ x[9:12]) + Ttau[1:4]
        )

        return ca.vertcat(dp, dv, deul, domega)

    def openloop_num(self, x, MF):
        Ttau = self.CoeffM_inv @ MF

        dp = x[3:6]

        dvx = (
            -Ttau[0]
            / self.m_actual
            * (np.cos(x[8]) * np.sin(x[7]) * np.cos(x[6]) + np.sin(x[8]) * np.sin(x[6]))
        )
        dvy = (
            -Ttau[0]
            / self.m_actual
            * (np.sin(x[8]) * np.sin(x[7]) * np.cos(x[6]) - np.cos(x[8]) * np.sin(x[6]))
        )
        dvz = 9.8 - Ttau[0] / self.m_actual * (np.cos(x[6]) * np.cos(x[7]))

        dv = (
            np.array([dvx, dvy, dvz])
            - self.aerodrag_num(x)
            + self.disturb[0:3] / self.m_actual
        )

        deul = (
            np.vstack(
                [
                    np.hstack(
                        [1, np.tan(x[7]) * np.sin(x[6]), np.tan(x[7]) * np.cos(x[6])]
                    ),
                    np.hstack([0, np.cos(x[6]), -np.sin(x[6])]),
                    np.hstack(
                        [0, np.sin(x[6]) / np.cos(x[7]), np.cos(x[6]) / np.cos(x[7])]
                    ),
                ]
            )
            @ x[9:12]
        )

        domega = self.J_actual_inv @ (
            -np.cross(x[9:12], self.J @ x[9:12]) + Ttau[1:4] + self.disturb[3:6]
        )

        return np.hstack([dp, dv, deul, domega])