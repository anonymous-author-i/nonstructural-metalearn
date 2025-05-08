import numpy as np
import scipy as sp

class L1AdaptiveEstimator:
    def __init__(self, nominal_model, output_select, adapt_gain, filter_gain, dt):
        self.nominal_model = nominal_model
        self.output_select = output_select
        self.dt = dt
        self.adapt_gain = -1.0 * adapt_gain
        self.filter_gain = filter_gain

        self.temp1 = sp.linalg.expm(self.adapt_gain * self.dt)
        self.temp2 = np.linalg.inv(self.temp1 - np.eye(self.adapt_gain.shape[0]))
        self.d_hat = np.zeros(len(output_select))

    def low_pass_dyn(self, x_f, x):
        dx_f = self.filter_gain @ (x - x_f)
        return dx_f

    def __esti_dyn(self, concat_hat, x_feedback, u):
        v_hat, d_hat = concat_hat[:len(self.output_select)], concat_hat[len(self.output_select):]
        dvhat = (
            self.nominal_model.openloop_num(x_feedback, u)[self.output_select]
            + d_hat
            + self.adapt_gain @ (v_hat - x_feedback[self.output_select])
        )

        d_new = (
            -self.temp2
            @ self.adapt_gain
            @ self.temp1
            @ (v_hat - x_feedback[self.output_select])
        )

        dd_hat = self.low_pass_dyn(d_hat, d_new)

        return np.hstack([dvhat, dd_hat])

    def d_esti_x(self, v_hat, x_feedback, u):
        h = self.dt
        d_hat = self.d_hat
        concat_hat = np.hstack([v_hat, d_hat])
        k1 = self.__esti_dyn(concat_hat, x_feedback, u)
        k2 = self.__esti_dyn((concat_hat + 0.5 * h * k1), x_feedback, u)
        k3 = self.__esti_dyn((concat_hat + 0.5 * h * k2), x_feedback, u)
        k4 = self.__esti_dyn((concat_hat + h * k3), x_feedback, u)
        concat_hat1 = concat_hat + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        v_hat1 = concat_hat1[:len(self.output_select)]
        d_hat1 = concat_hat1[len(self.output_select):]
        self.d_hat = d_hat1
        return v_hat1, d_hat1
