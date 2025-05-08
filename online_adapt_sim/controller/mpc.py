import numpy as np
import casadi as ca


class MPC:
    def __init__(self, model, discrete_h, H):
        self.model = model
        self.h = discrete_h
        self.H = H
        # Quadratic Objectives
        self.Q_k = 1e2 * np.diag(
            np.array([1.0] * 3 + [0.5] * 3 + [0.5] * 3 + [0.01] * 3)
        )
        self.R_k = np.diag(np.array([1.0] * 4))
        self.Q_f = self.Q_k
        # State Bound
        self.x_lb = [-ca.inf] * 6 + [-np.pi / 2] * 3 + [-ca.inf] * 3
        self.x_ub = [ca.inf] * 6 + [np.pi / 2] * 3 + [ca.inf] * 3
        # Input Bound
        self.u_lb = list(self.model.u_lb)
        self.u_ub = list(self.model.u_ub)

        # NLP Setup
        self.build_symbolic()

    def loss_kx(self, xk, xk_ref):
        return 0.5 * ca.transpose(xk - xk_ref) @ self.Q_k @ (xk - xk_ref)

    def loss_ku(self, uk, uk_ref):
        return 0.5 * ca.transpose(uk - uk_ref) @ self.R_k @ (uk - uk_ref)

    def loss_k(self, xk, uk, xk_ref, uk_ref):
        return self.loss_kx(xk, xk_ref) + self.loss_ku(uk, uk_ref)

    def loss_N(self, xf, xf_ref):
        return self.loss_kx(xf, xf_ref)

    def set_stateBoxCons(self, lb_list, ub_list):
        self.x_lb = lb_list
        self.x_ub = ub_list

    def set_inputBoxCons(self, lb_list, ub_list):
        self.u_lb = lb_list
        self.u_ub = ub_list

    def discrete_sys(self, xk, uk):
        h = self.h
        k1 = self.model.openloop_sym(xk, uk)
        k2 = self.model.openloop_sym((xk + 0.5 * h * k1), uk)
        k3 = self.model.openloop_sym((xk + 0.5 * h * k2), uk)
        k4 = self.model.openloop_sym((xk + h * k3), uk)
        xk1 = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1

    def build_symbolic(self):
        x = ca.SX.sym("x", 12)
        u = ca.SX.sym("u", 4)
        rhs = self.discrete_sys(x, u)
        self.discrete_dynfunc = ca.Function("dyn", [x, u], [rhs])
        x_ref = ca.SX.sym("x_ref", 12)
        u_ref = ca.SX.sym("u_ref", 4)
        loss_k = self.loss_k(x, u, x_ref, u_ref)
        loss_N = self.loss_N(x, x_ref)
        self.lk = ca.Function("lkx", [x, u, x_ref, u_ref], [loss_k])
        self.lN = ca.Function("lku", [x, x_ref], [loss_N])

    def solve_OC(self, x0, xref_seq, uref_seq):
        '''
        Solve u with Nominal State

        refk_seq: comming up reference trajectory: state_dim * H

        nonlinear program (NLP):
            min          F(x, p)
            x

            subject to
            LBX <=   x    <= UBX
            LBG <= G(x, p) <= UBG
            p  == P

            nx: number of decision variables
            ng: number of constraints
            np: number of parameters
        '''

        w = []  # Solution
        w0 = []  # Init guess
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # Optimize N*states+N*inputs = 16*N parameters
        Xk_ = ca.DM(x0)  # The first states
        # H = xref_seq.shape[1]
        for k in range(self.H):
            # Add control param
            Uk = ca.MX.sym("U_" + str(k), self.model.nomi_u_dim)
            w += [Uk]
            w0 += list(uref_seq[:, k])
            lbw += self.u_lb
            ubw += self.u_ub

            # Add state param and constraint (index + 1)
            Xk1 = ca.MX.sym("X_" + str(k + 1), self.model.nomi_x_dim)
            w += [Xk1]
            w0 += list(xref_seq[:, k])
            lbw += self.x_lb
            ubw += self.x_ub

            # Add continous constraint
            g += [self.discrete_dynfunc(Xk_, Uk) - Xk1]
            lbg += [0] * self.model.x_dim
            ubg += [0] * self.model.x_dim

            # Add Obj
            if k == self.H - 1:
                J += self.lk(Xk_, Uk, xref_seq[:, k], uref_seq[:, k])
            else:
                J += self.lN(Xk_, xref_seq[:, k])

            # Refresh Xk_
            Xk_ = Xk1

        # Build Prob
        prob = {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}
        # options = {}
        options = {
            "verbose": False,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False,
        }
        solver = ca.nlpsol("solver", "ipopt", prob, options)

        # Solve
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # Get control and state in mpc
        result = (
            sol["x"]
            .full()
            .flatten()
            .reshape(-1, self.model.x_dim + self.model.u_dim)
        )
        result = result.T
        u_opt_seq = result[0 : self.model.u_dim, :]
        x_opt_seq = result[self.model.u_dim :, :]

        return u_opt_seq, x_opt_seq

    def dx_modelpredict(self, xk, uk):
        dx = self.model.openloop_forCtrl(xk, uk)
        return dx


class AdaptMPC:
    def __init__(self, model, discrete_h, H):
        self.model = model
        self.h = discrete_h
        self.H = H
        # Quadratic Objectives
        self.Q_k = 1e2 * np.diag(
            np.array([1.0] * 3 + [0.5] * 3 + [0.5] * 3 + [0.01] * 3)
        )
        self.R_k = np.diag(np.array([1.0] * 4))
        self.Q_f = self.Q_k
        # State Bound
        self.x_lb = [-ca.inf] * 6 + [-np.pi / 2] * 3 + [-ca.inf] * 3
        self.x_ub = [ca.inf] * 6 + [np.pi / 2] * 3 + [ca.inf] * 3
        # Input Bound
        self.u_lb = list(self.model.u_lb)
        self.u_ub = list(self.model.u_ub)

        # NLP Setup
        self.build_symbolic()

    def loss_kx(self, xk, xk_ref):
        return 0.5 * ca.transpose(xk - xk_ref) @ self.Q_k @ (xk - xk_ref)

    def loss_ku(self, uk, uk_ref):
        return 0.5 * ca.transpose(uk - uk_ref) @ self.R_k @ (uk - uk_ref)

    def loss_k(self, xk, uk, xk_ref, uk_ref):
        return self.loss_kx(xk, xk_ref) + self.loss_ku(uk, uk_ref)

    def loss_N(self, xf, xf_ref):
        return self.loss_kx(xf, xf_ref)

    def set_stateBoxCons(self, lb_list, ub_list):
        self.x_lb = lb_list
        self.x_ub = ub_list

    def set_inputBoxCons(self, lb_list, ub_list):
        self.u_lb = lb_list
        self.u_ub = ub_list

    def discrete_sys(self, xk, uk, ff_inputs, FC_vec):
        h = self.h
        k1 = self.model.openloop_sym(xk, uk, ff_inputs, FC_vec)
        k2 = self.model.openloop_sym((xk + 0.5 * h * k1), uk, ff_inputs, FC_vec)
        k3 = self.model.openloop_sym((xk + 0.5 * h * k2), uk, ff_inputs, FC_vec)
        k4 = self.model.openloop_sym((xk + h * k3), uk, ff_inputs, FC_vec)
        xk1 = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1

    def build_symbolic(self):
        x = ca.SX.sym("x", 12)
        u = ca.SX.sym("u", 4)
        ff_inputs = ca.SX.sym(
            "ffin",
            self.model.predictor.state_dim + self.model.predictor.control_dim,
            self.model.predictor.history_len + 1,
        )
        FC_vec = ca.SX.sym(
            "FC_vec",
            self.model.predictor.output_dim * self.model.predictor.hidden_sizes[-1],
        )
        rhs = self.discrete_sys(x, u, ff_inputs, FC_vec)
        self.discrete_dynfunc = ca.Function("dyn", [x, u, ff_inputs, FC_vec], [rhs])
        x_ref = ca.SX.sym("x_ref", 12)
        u_ref = ca.SX.sym("u_ref", 4)
        loss_k = self.loss_k(x, u, x_ref, u_ref)
        loss_N = self.loss_N(x, x_ref)
        self.lk = ca.Function("lkx", [x, u, x_ref, u_ref], [loss_k])
        self.lN = ca.Function("lku", [x, x_ref], [loss_N])

    def refresh(self, ff_inputs_init):
        # clear the memory for ff_inputs
        self.ff_inputs = ff_inputs_init

    def update_ffinputs(self, ffinput):
        self.ff_inputs = ca.horzcat(self.ff_inputs, ffinput)
        self.ff_inputs = self.ff_inputs[:, :-1]

    def load_FCvec(self, FCvec):
        self.FCvec = FCvec

    def solve_OC(self, x0, xref_seq, uref_seq):
        '''
        Solve u with Nominal State

        refk_seq: comming up reference trajectory: state_dim * H

        nonlinear program (NLP):
            min          F(x, p)
            x

            subject to
            LBX <=   x    <= UBX
            LBG <= G(x, p) <= UBG
            p  == P

            nx: number of decision variables
            ng: number of constraints
            np: number of parameters
        '''

        w = []  # Solution
        w0 = []  # Init guess
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # Optimize N*states+N*inputs = 16*N parameters
        Xk_ = ca.DM(x0)  # The first states
        # H = xref_seq.shape[1]
        for k in range(self.H):
            # Add control param
            Uk = ca.MX.sym("U_" + str(k), self.model.u_dim)
            w += [Uk]
            w0 += list(uref_seq[:, k])
            lbw += self.u_lb
            ubw += self.u_ub

            # Add state param and constraint (index + 1)
            Xk1 = ca.MX.sym("X_" + str(k + 1), self.model.x_dim)
            w += [Xk1]
            w0 += list(xref_seq[:, k])
            lbw += self.x_lb
            ubw += self.x_ub

            # Add continous constraint
            g += [
                self.discrete_dynfunc(Xk_, Uk, self.ff_inputs, self.FCvec) - Xk1
            ]
            lbg += [0] * self.model.x_dim
            ubg += [0] * self.model.x_dim

            # Add Obj
            if k == self.H - 1:
                J += self.lk(Xk_, Uk, xref_seq[:, k], uref_seq[:, k])
            else:
                J += self.lN(Xk_, xref_seq[:, k])

            # Refresh Xk_
            Xk_ = Xk1

            # Update ff_inputs
            self.update_ffinputs(Xk_[self.model.predictor.state_select])
 

        # Build Prob
        prob = {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}
        # options = {}
        options = {
            "verbose": False,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False,
        }
        solver = ca.nlpsol("solver", "ipopt", prob, options)

        # Solve
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # Get control and state in mpc
        result = (
            sol["x"]
            .full()
            .flatten()
            .reshape(-1, self.model.x_dim + self.model.u_dim)
        )
        result = result.T
        u_opt_seq = result[0 : self.model.u_dim, :]
        x_opt_seq = result[self.model.u_dim :, :]

        return u_opt_seq, x_opt_seq
