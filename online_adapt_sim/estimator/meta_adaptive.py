import numpy as np


class MetaAdaptiveEstimator:
    def __init__(
        self, ff_model, ff_enable, adapt_gain, feedback_gain, concurrent_learn_len
    ):
        # Setups
        self.ff_model = ff_model
        self.state_dim = self.ff_model.model_n.nomi_x_dim
        self.stack_len = self.ff_model.predictor.history_len + 1
        self.feedback_gain = feedback_gain
        self.adapt_gain_init = adapt_gain
        self.concurrent_learn_len = concurrent_learn_len
        self.ff_enable = ff_enable

        self.beta0 = 20.0
        self.bound = 5.0

    """ for learning """

    def init_learner_and_stack(self, init_ffinput, init_dx):
        self.ff_inputs = np.hstack([init_ffinput] * self.stack_len)
        self.seq_fill_id = 0
        self.dx_seq = np.hstack([init_dx] * self.concurrent_learn_len)
        self.disturb_seq = np.hstack(
            [np.zeros((len(self.ff_model.predictor.output_select), 1))]
            * self.concurrent_learn_len
        )
        self.ffinputs_stack = np.dstack([self.ff_inputs] * self.concurrent_learn_len)
        self.stack_fill_id = 0
        self.adapt_gain = self.adapt_gain_init
        self.adapt_gain_inv = np.linalg.inv(self.adapt_gain)

    def init_learner(self):
        self.adapt_gain = self.adapt_gain_init
        self.adapt_gain_inv = np.linalg.inv(self.adapt_gain)

    def store_data(self, ffinput, dx, disturb):
        # store current ff_input, dx and x, update the stack for concurrent learning
        if self.seq_fill_id < self.stack_len:
            self.ff_inputs[:, self.seq_fill_id : self.seq_fill_id + 1] = ffinput
            self.seq_fill_id += 1
        else:
            self.ff_inputs[:, :-1] = self.ff_inputs[:, 1:]
            self.ff_inputs[:, -2:-1] = ffinput

        self.update_ffinputs(self.ff_inputs, dx, disturb)

    def update_ffinputs(self, ffinputs, dx, disturb):
        # update current ff_inputs, dx, disturbance store into the stack of ff_inputs
        if self.stack_fill_id < self.concurrent_learn_len:
            self.ffinputs_stack[:, :, self.stack_fill_id] = ffinputs
            self.dx_seq[:, self.stack_fill_id : self.stack_fill_id + 1] = dx
            self.disturb_seq[:, self.stack_fill_id : self.stack_fill_id + 1] = disturb
            self.stack_fill_id += 1
        else:
            self.ffinputs_stack[:, :, :-1] = self.ffinputs_stack[:, :, 1:]
            self.ffinputs_stack[:, :, -1] = ffinputs
            self.dx_seq[:, :-1] = self.dx_seq[:, 1:]
            self.dx_seq[:, -2:-1] = dx
            self.disturb_seq[:, :-1] = self.disturb_seq[:, 1:]
            self.disturb_seq[:, -2:-1] = disturb

    def __grad_adapt(self, ff_inputs, err, h, Theta):
        # gradient adaptation law
        dtheta = (
            self.adapt_gain
            @ self.ff_model.predictor.basis_func(ff_inputs).T
            @ (err[self.ff_model.predictor.output_select])
        ) - 0.001 * self.adapt_gain @ Theta
        return Theta + dtheta * h

    def __gain_update(self, Ginv, G, ff_inputs, h):
        # gain update for recurrent least square law
        beta = self.beta0 * (1 - np.linalg.norm(G) / self.bound)
        y = self.ff_model.predictor.basis_func(ff_inputs)
        dGinv = -beta * Ginv + y.T @ y
        return dGinv * h

    def rls_learn(self, x_feedback, dx_feedback, u, Theta):
        # recurrent least square law
        ff_inputs = self.ff_inputs
        h = self.ff_model.h
        dx_pred = self.ff_model.openloop_num(x_feedback, u, ff_inputs, Theta)
        dx_feedback = np.squeeze(dx_feedback, axis=1)
        Theta1 = self.__grad_adapt(ff_inputs, dx_feedback - dx_pred, h, Theta)
        self.adapt_gain = np.linalg.inv(self.adapt_gain_inv)
        self.adapt_gain_inv += self.__gain_update(
            self.adapt_gain_inv, self.adapt_gain, ff_inputs, h
        )
        return Theta1

    def concurrent_learn(self, Theta):
        # concurrent learning
        concurrent_loss = 0
        ff_inputs_stack = self.ffinputs_stack
        disturb_seq = self.disturb_seq
        for idx in range(ff_inputs_stack.shape[2]):
            ff_inputs = ff_inputs_stack[:, :, idx]
            disturb = disturb_seq[:, idx]
            err = disturb - self.ff_model.predictor.predict(ff_inputs, Theta)
            concurrent_loss += self.ff_model.predictor.basis_func(ff_inputs).T @ err
        dtheta = self.adapt_gain_init @ concurrent_loss
        return Theta + dtheta * self.ff_model.h

    def concurrent_learn(self,Theta):
        # concurrent learning
        concurrent_loss = 0
        ff_inputs_stack = self.ffinputs_stack
        disturb_seq = self.disturb_seq
        for idx in range(ff_inputs_stack.shape[2]):
            ff_inputs = ff_inputs_stack[:, :, idx]
            err = disturb_seq[:, idx] - self.ff_model.predictor.predict(ff_inputs, Theta)
            concurrent_loss += self.ff_model.predictor.basis_func(ff_inputs).T @ (err)
        dtheta = self.adapt_gain_init @ concurrent_loss
        return Theta + dtheta * self.ff_model.h

    def concurrent_learn_composite(self, track_err, Theta):
        # concurrent learning
        concurrent_loss = 0
        ff_inputs_stack = self.ffinputs_stack
        disturb_seq = self.disturb_seq
        for idx in range(ff_inputs_stack.shape[2]):
            ff_inputs = ff_inputs_stack[:, :, idx]
            err = disturb_seq[:, idx] - self.ff_model.predictor.predict(ff_inputs, Theta)
            concurrent_loss += self.ff_model.predictor.basis_func(ff_inputs).T @ (err)
        dtheta = (
            self.adapt_gain_init @ concurrent_loss
            + self.adapt_gain_init
            @ self.ff_model.predictor.basis_func(ff_inputs).T
            @ track_err
        )
        return Theta + dtheta * self.ff_model.h

    def direct_least_square(self, reg=0.01):
        ff_inputs_stack = self.ffinputs_stack[:, :, :-1]
        disturb_seq = self.disturb_seq[:, :-1]
        basis_vec = []
        y_vec = []
        for idx in range(ff_inputs_stack.shape[2]):
            basis_vec += [self.ff_model.predictor.basis_func(ff_inputs_stack[:, :, idx])]
            y_vec += [disturb_seq[:, idx:idx+1]]
        basis_vec = np.vstack(basis_vec)
        y_vec = np.vstack(y_vec)
        Theta = (
            np.linalg.inv(basis_vec.T @ basis_vec + np.eye(self.adapt_gain_init.shape[0]) * reg)
            @ basis_vec.T
            @ y_vec
        )
        # pred = self.ff_model.predictor.basis_func(ff_inputs_stack[:, :, -1]) @ Theta
        return np.squeeze(Theta, axis=1)


    """ for estimation """

    # use low-pass filter on estimation err
    def __d_esti_dyn_x(self, z_hat, x_feedback, u, ff_inputs, Theta):
        if self.ff_enable:
            pred = self.ff_model.predictor.predict(ff_inputs, Theta)
        else:
            pred = np.zeros(len(self.ff_model.predictor.output_select))
        dz_hat = -self.feedback_gain * (
            z_hat
            + pred
            + self.feedback_gain * x_feedback[self.ff_model.predictor.output_select]
            + self.ff_model.model_n.openloop_num(x_feedback, u)[
                self.ff_model.predictor.output_select
            ]
        )
        d_hat = (
            z_hat
            + pred
            + self.feedback_gain * x_feedback[self.ff_model.predictor.output_select]
        )
        return dz_hat, d_hat

    def __d_esti_dyn_dx(self, z_hat, dx_feedback, x_feedback, u, ff_inputs, Theta):
        if self.ff_enable:
            pred = self.ff_model.predictor.predict(ff_inputs, Theta)
        else:
            pred = np.zeros(len(self.ff_model.predictor.output_select))
        disturb = dx_feedback - self.ff_model.model_n.openloop_num(x_feedback, u)
        dz_hat = self.feedback_gain * (
            disturb[self.ff_model.predictor.output_select] - z_hat - pred
        )
        d_hat = z_hat + pred
        return dz_hat, d_hat

    def d_esti_x(self, z_hat, x_feedback, u, Theta):
        ff_inputs = self.ff_inputs
        h = self.ff_model.h
        k1, d1 = self.__d_esti_dyn_x(z_hat, x_feedback, u, ff_inputs, Theta)
        k2, _ = self.__d_esti_dyn_x(
            (z_hat + 0.5 * h * k1), x_feedback, u, ff_inputs, Theta
        )
        k3, _ = self.__d_esti_dyn_x(
            (z_hat + 0.5 * h * k2), x_feedback, u, ff_inputs, Theta
        )
        k4, _ = self.__d_esti_dyn_x((z_hat + h * k3), x_feedback, u, ff_inputs, Theta)
        z_hat1 = z_hat + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        return z_hat1, d1

    def d_esti_dx(self, z_hat, dx_feedback, x_feedback, u, Theta):
        ff_inputs = self.ff_inputs
        h = self.ff_model.h
        k1, d1 = self.__d_esti_dyn_dx(
            z_hat, dx_feedback, x_feedback, u, ff_inputs, Theta
        )
        k2, _ = self.__d_esti_dyn_dx(
            (z_hat + 0.5 * h * k1), dx_feedback, x_feedback, u, ff_inputs, Theta
        )
        k3, _ = self.__d_esti_dyn_dx(
            (z_hat + 0.5 * h * k2), dx_feedback, x_feedback, u, ff_inputs, Theta
        )
        k4, _ = self.__d_esti_dyn_dx(
            (z_hat + h * k3), dx_feedback, x_feedback, u, ff_inputs, Theta
        )
        z_hat1 = z_hat + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        return z_hat1, d1
