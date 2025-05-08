import numpy as np
import casadi as ca
import torch
from utils import *


class PredictorShell_Adapt:
    def __init__(self, settings, torch_model, mode):
        '''
        A shell of predictor models for numerical simulation and optimal control
        mode = "normal" (for numerical simulation) or "OC" (for optimal control)
        '''
        self.mode = mode
        if self.mode != "normal" and self.mode != "OC":
            raise AttributeError(
                "mode: 'normal' for numerical simulation or 'OC' for optimal control"
            )

        self.state_select = settings["state_select"]
        self.control_select = settings["control_select"]
        self.output_select = settings["output_select"]
        self.state_dim = len(self.state_select)
        self.control_dim = len(self.control_select)
        self.output_dim = len(self.output_select)
        self.history_len = settings["history_len"]
        self.input_size = (len(self.state_select) + len(self.control_select)) * (
            self.history_len + 1
        )
        self.hidden_sizes = settings["hidden_sizes"]

        self.torch_model = torch_model
        self.load_model_params(self.torch_model.parameters())

    def load_model_params(self, model_param_gen):
        # Pytorch model parameter to a vector
        params = []
        for param in model_param_gen:
            param_cpu = param.cpu()
            param_array = param_cpu.detach().numpy().reshape(-1, 1)
            params += [param_array]
        self.params = np.vstack(params)

    def __linear_layer(self, x, param, dim_in, dim_out):
        W = param[0 : dim_in * dim_out].reshape((dim_out, dim_in))
        b = param[dim_in * dim_out : dim_in * dim_out + dim_out]
        # print(W.shape, x.shape, b.shape)
        return W @ x + b

    def MLP_withoutFC(self, x):
        params = self.params

        x_ = x.reshape((-1, 1))
        param_idx = 0
        param_idx_1 = 0
        # Input Layer
        dim_in = self.input_size
        dim_out = self.hidden_sizes[0]
        param_idx_1 += dim_in * dim_out + dim_out
        param_ = params[param_idx:param_idx_1]
        param_idx = param_idx_1

        x_ = self.__linear_layer(x_, param_, dim_in, dim_out)
        x_ = np.tanh(x_)

        # Hidden Layer 1
        dim_in = self.hidden_sizes[0]
        dim_out = self.hidden_sizes[1]
        param_idx_1 += dim_in * dim_out + dim_out
        param_ = params[param_idx:param_idx_1]
        param_idx = param_idx_1

        x_ = self.__linear_layer(x_, param_, dim_in, dim_out)
        x_ = np.tanh(x_)

        # # Hidden Layer 2
        # dim_in = self.hidden_sizes[1]
        # dim_out = self.hidden_sizes[2]
        # param_idx_1 += dim_in * dim_out + dim_out
        # param_ = params[param_idx:param_idx_1]
        # param_idx = param_idx_1

        # x_ = self.__linear_layer(x_, param_, dim_in, dim_out)
        # x_ = np.tanh(x_)

        # # Hidden Layer 3
        # dim_in = self.hidden_sizes[2]
        # dim_out = self.hidden_sizes[2]
        # param_idx_1 += dim_in * dim_out + dim_out
        # param_ = params[param_idx:param_idx_1]
        # param_idx = param_idx_1

        # x_ = self.__linear_layer(x_, param_, dim_in, dim_out)
        # x_ = np.tanh(x_)

        self.params_base_dim = param_idx_1

        return x_

    def basis_func(self, ff_inputs):
        if self.mode == "normal":
            ff_inputs = torch.tensor(ff_inputs, dtype=torch.float)
            basis_output = self.torch_model.forward(ff_inputs).detach().reshape((-1, 1))
            out = torch.block_diag(*[basis_output] * self.output_dim)
            return out.numpy().T
        elif self.mode == "OC":
            x = ff_inputs.reshape((-1, 1))
            basis_output = self.MLP_withoutFC(x)
            return ca.diagcat(*[basis_output] * self.output_dim).T

    def predict(self, ff_inputs, Theta):
        out = self.basis_func(ff_inputs) @ Theta
        return out
    

class PredictorShell_Neural:
    def __init__(self, params, state_select, output_select, nn_input_size=6, nn_hidden_size=36, nn_output_size=3):
        self.params = params
        self.state_select = state_select
        self.output_select = output_select
        self.input_size = nn_input_size
        self.hidden_size = nn_hidden_size
        self.output_size = nn_output_size

    def __linear_layer(self, x, param, dim_in, dim_out):
        W = param[0 : dim_in * dim_out].reshape((dim_out, dim_in))
        b = param[dim_in * dim_out : dim_in * dim_out + dim_out]
        # print(W.shape, x.shape, b.shape)
        return W @ x + b
    
    def __linear_layer_nobias(self, x, param, dim_in, dim_out):
        W = param[0 : dim_in * dim_out].reshape((dim_out, dim_in))
        return W @ x
    
    def MLP(self, x):
        params = self.params

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

    def predict(self, x):
        # neural-network predicts drag (a negative residual)
        out = -self.MLP(x[self.state_select])
        return np.squeeze(out, axis=1)

class Model_Wrapper:
    def __init__(self, nominal_model, predictor_model, discrete_h):
        '''
        Model Wrapper for numerical simulation and optimal control problems

        Wraps up nominal model and learning basis for online learning.

        '''
        # Wrap up nominal model and residual model
        self.model_n = nominal_model
        self.predictor = predictor_model

        self.h = discrete_h

        # Define model dimensions
        self.x_dim = self.model_n.nomi_x_dim
        self.u_dim = self.model_n.nomi_u_dim

        # Define model control bounds
        self.u_lb = self.model_n.u_lb
        self.u_ub = self.model_n.u_ub
         

    '''Continuous Model'''

    # define augmented dynamics
    def openloop_sym(self, x, u, *args):
        dx = self.model_n.openloop_sym(x, u)
        if self.predictor:
            dx[self.predictor.output_select] += self.predictor.predict(*args)
            print("neural_on")
        return dx

    def openloop_num(self, x, u, *args):
        dx = self.model_n.openloop_num(x, u)
        if self.predictor:
            dx[self.predictor.output_select] += self.predictor.predict(*args)
        return dx


class ClosedLoop_Wrapper:
    def __init__(
        self,
        openloop_dynamics,
        controller,
        discrete_h,
        ffcomp_on,
    ):
        self.sim_model = openloop_dynamics
        self.controller = controller
        self.h = discrete_h
        self.ffcomp_on = ffcomp_on

        self.x_dim = self.sim_model.x_dim
        self.ctrl_slackvar_dim = self.controller.ctrl_slackvar_dim
        self.u = np.zeros(self.sim_model.u_dim)
        self.a_des = np.zeros(self.controller.pos_out_dim)

    def closedloop_num(self, concat_state, refx, d_hat, disturb=0.0):
        x, aux = (
            concat_state[0 : self.x_dim],
            concat_state[-self.ctrl_slackvar_dim :],
        )
        if self.ffcomp_on:
            u, daux, a_des, eul_des, omega_des = self.controller.ctrlmap_num(x, refx, aux, d_hat)
        else:
            u, daux, a_des, eul_des, omega_des = self.controller.ctrlmap_num(x, refx, aux)
        dx = self.sim_model.openloop_num(x, u)
        # store control info
        self.u = u  
        self.a_des = a_des
        self.eul_des = eul_des
        self.omega_des = omega_des
        return np.hstack([dx, daux])

    def closedloop_num_exRK4(self, concat_state, refx, d_hat):
        h = self.h
        k1 = self.closedloop_num(concat_state, refx, d_hat)
        k2 = self.closedloop_num((concat_state + 0.5 * h * k1), refx, d_hat)
        k3 = self.closedloop_num((concat_state + 0.5 * h * k2), refx, d_hat)
        k4 = self.closedloop_num((concat_state + h * k3), refx, d_hat)
        concat_state1 = concat_state + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return concat_state1
