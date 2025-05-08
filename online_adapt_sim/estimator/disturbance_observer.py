import numpy as np


class DisturbanceObserver:
    def __init__(self, ff_model, output_select, feedback_gain, dt):
        self.ff_model = ff_model
        self.output_select = output_select
        self.feedback_gain = feedback_gain
        self.dt = dt

    def __d_esti_dyn_x(self, z_hat, x_feedback, u):
        if self.ff_model.predictor:
            pred = self.ff_model.predictor.predict(x_feedback)
        else:
            pred = np.zeros(len(self.output_select))
        dz_hat = -self.feedback_gain @ (
            z_hat
            + pred
            + self.feedback_gain @ x_feedback[self.output_select]
            + self.ff_model.model_n.openloop_num(x_feedback, u)[self.output_select]
        )
        d_hat = (
            z_hat
            + pred
            + self.feedback_gain @ x_feedback[self.output_select]
        )
        return dz_hat, d_hat
    
    def d_esti_x(self, z_hat, x_feedback, u):
        h = self.dt
        k1, d1 = self.__d_esti_dyn_x(z_hat, x_feedback, u)
        k2, _ = self.__d_esti_dyn_x((z_hat + 0.5 * h * k1), x_feedback, u)
        k3, _ = self.__d_esti_dyn_x((z_hat + 0.5 * h * k2), x_feedback, u)
        k4, _ = self.__d_esti_dyn_x((z_hat + h * k3), x_feedback, u)
        z_hat1 = z_hat + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        return z_hat1, d1
