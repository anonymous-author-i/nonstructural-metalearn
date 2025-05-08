import numpy as np
import casadi as ca
import random


"""
Trajectory Utils:
1. [class] Polynomial Trajectory Optimizaion Solver
2. [class] Polynomial Trajectory Generator with Randomization Seetings
3. [function] coeff_2_pointsPoly:
    computes a sequence of discrete point from polynomial coefficents
4. [function] coeff_2_PointsLissajous:
    computes a sequence of discrete points from Lissajous trajectories with analytic expression
"""

class Polynomial_TrajGen:
    """
    A trajectory optimization solver for differential flatness systems
    outputs: p, v, a, j
    The trajectory is defined in polynomial form but each discrete
    point is optimized with the timestep between them fixed, 
    instead of optimizing polynimial coeffcients.

    Here we import the flat dynamics as well as other
    mappings from a quadrotor model.
    The output result is the input command of the controller (see DFBC/ Mellinger Controller).
    """

    def __init__(self, model):
        self.model = model

    """Problem Setup"""

    def discrete_setup(self, N, h):
        self.N = N
        self.h = h

    # def set_stateBoxCons(self, X_lb_list, X_ub_list):
    #     self.X_lb_list = X_lb_list
    #     self.X_ub_list = X_ub_list

    def set_refuBoxCons(self, inputlb, inputub):
        # Box constraints on reference u (snap)
        self.inputlb = inputlb
        self.inputub = inputub

    def set_modeluBoxCons(self, model_u_lb, model_u_ub):
        # Box constraints on model u
        self.model_u_lb = model_u_lb
        self.model_u_ub = model_u_ub

    def set_refxBoundCond(self, x0, xf):
        self.x0 = x0
        self.xf = xf

    def set_posWP(self, posWPs):
        self.posWPs = posWPs

    def get_lspaceLineSeg(self, posWPs):
        wp_Xfull = posWPs
        dis_list = [
            np.linalg.norm(wp_Xfull[i] - wp_Xfull[i + 1])
            for i in range(wp_Xfull.shape[0] - 1)
        ]
        total_dis = sum(dis_list)

        NiSeg_list = [
            int(dis_list[i] / total_dis * self.N)
            for i in range(wp_Xfull.shape[0] - 1 - 1)
        ]
        NiSeg_list += [self.N - sum(NiSeg_list)]
        Ni_list = [sum(NiSeg_list[:i]) for i in range(len(NiSeg_list))] + [self.N]

        LineSeg = [
            np.linspace(wp_Xfull[i], wp_Xfull[i + 1], NiSeg_list[i], endpoint=False)
            for i in range(len(Ni_list) - 1)  # 1 less than the length
        ] + [wp_Xfull[-1]]

        LineSeg = np.vstack(LineSeg)
        return LineSeg, Ni_list[1:]

    def NLP_Prepare(self):
        # Perpare System and Mappings
        ref_states = ca.SX.sym("ref_states", self.model.refx_dim)
        ref_inputs = ca.SX.sym("ref_inputs", self.model.refu_dim)
        ref_rhs = self.model.refsys_RK4(ref_states, ref_inputs)

        state_refout = self.model.ref2x_map(ref_states)
        input_refout = self.model.ref2u_map(ref_states)

        self.refdyn_RK4 = ca.Function("refdyn_RK4", [ref_states, ref_inputs], [ref_rhs])
        self.ref2x_map = ca.Function("ref2x_map", [ref_states], [state_refout])
        self.ref2u_map = ca.Function("ref2u_map", [ref_states], [input_refout])

    def NLP_FormAndSolve(self, Eq_Relax):
        """
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
        """

        w = []  # Solution
        w0 = []  # Init guess
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # Waypoints and Related
        # uk_ref = np.array([self.m*9.8/4]*4 + [0.01])
        LineSeg, Ni_list = self.get_lspaceLineSeg(self.posWPs)
        # WP_loc = [0] + Ni_list

        refXk_ = ca.DM(
            np.hstack([self.posWPs[0, :], np.zeros(9)])
        )  # Previous flat States

        for k in range(self.N):
            """Add Snap, time param and constraint"""
            refUk_except_h = ca.MX.sym("S_" + str(k), self.model.refu_dim - 1)
            w += [refUk_except_h]
            w0 += list([0] * 3)
            lbw += self.inputlb
            ubw += self.inputub

            """Add flat-state param and constraint (index + 1)"""
            refXk1 = ca.MX.sym("X_" + str(k + 1), self.model.refx_dim)
            w += [refXk1]
            w0 += list(np.hstack([LineSeg[k, 0:3], np.zeros(9)]))
            # w0 += [0]*12
            lbw += [-ca.inf] * self.model.refx_dim
            ubw += [ca.inf] * self.model.refx_dim

            # Add continous constraint
            g += [self.refdyn_RK4(refXk_, ca.vertcat(refUk_except_h, self.h)) - refXk1]
            lbg += [0] * self.model.refx_dim
            ubg += [0] * self.model.refx_dim

            # Add mfs constraints
            g += [self.ref2u_map(refXk_)]
            lbg += self.model_u_lb
            ubg += self.model_u_ub

            """Add Position Waypoint and Terminal Constraints"""
            if k + 1 in Ni_list:
                if k + 1 == self.N:
                    g += [refXk1 - self.xf]
                    lbg += [-Eq_Relax] * self.model.refx_dim
                    ubg += [Eq_Relax] * self.model.refx_dim
                else:
                    g += [refXk1[0:3] - LineSeg[k + 1, :]]
                    lbg += [-Eq_Relax] * 3
                    ubg += [Eq_Relax] * 3

            # Refresh flatXk_
            refXk_ = refXk1

        # Build Prob
        prob = {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}
        # options = {"ipopt.hessian_approximation" : "limited-memory"}
        options = {
            "verbose": False,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 5,
            "print_time": True,
            # "ipopt.hessian_approximation":"limited-memory"
        }
        solver = ca.nlpsol("solver", "ipopt", prob, options)

        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        return sol, solver.stats()["success"]
    
    def get_refxopt(self, sol):
        # Demux Solution
        result = (
            sol["x"]
            .full()
            .flatten()
            .reshape(-1, self.model.refx_dim + self.model.refu_dim - 1)
        )
        refx_opt = result[:-1, self.model.refu_dim - 1 :]
        return refx_opt


class Polynomial_TrajGen(Polynomial_TrajGen):
    """
    A trajectory optimization generator for differential flatness systems
    outputs: p, v, a, j
    Built upon the solver (Polynomial_TrajOpt) with random setups of optimization parameters.
    """
    def __init__(self, model):
        super(Polynomial_TrajGen, self).__init__(model)

    def set_rand_waypoints(self, max_waypoint_num, x_bound, y_bound, z_bound, rand_seed=None):
        if rand_seed:
            np.random.seed(rand_seed)
        waypoints = []
        waypoint_num = np.random.randint(2, max_waypoint_num)
        for _ in range(waypoint_num):
            x = np.random.uniform(x_bound[0], x_bound[1])
            y = np.random.uniform(y_bound[0], y_bound[1])
            z = np.random.uniform(z_bound[0], z_bound[1])
            waypoints += [np.array([x, y, z])]
        
        self.posWPs = np.vstack(waypoints)
        return self.posWPs

    def set_rand_discreteN(self, h, N_min, N_max, rand_seed=None):
        if rand_seed:
            np.random.seed(rand_seed)
        self.h = h
        self.N = np.random.randint(N_min, N_max)

        return self.N
    
    def set_rand_refuBoxCons(self, max_refu_mag, rand_seed=None):
        if rand_seed:
            np.random.seed(rand_seed)
        bound = max_refu_mag * np.random.rand(1)
        self.inputlb = [-bound] * (self.model.refu_dim - 1)
        self.inputub = [bound] * (self.model.refu_dim - 1)

        return bound

    def set_rand_modeluBoxCons(self, min_u, max_u, rand_seed=None):
        if rand_seed:
            np.random.seed(rand_seed)
        bound = np.random.uniform(min_u, max_u)
        self.model_u_lb = [min_u] * self.model.nomi_u_dim
        self.model_u_ub = [bound] * self.model.nomi_u_dim

        return bound

def coeff_to_pointsLissajous(t, a, a0, Radi, Period, h):
    t = a0 * t
    a = a0 * a
    rx, ry, rz = Radi
    Tx, Ty, Tz = Period
    wx, wy, wz = 2 * np.pi / Tx, 2 * np.pi / Ty, 2 * np.pi / Tz

    P = np.array(
        [rx * np.sin(wx * t), ry * np.sin(wy * t), h + rz * np.cos(wz * t)]
    )
    V = a * np.array(
        [
            wx * rx * np.cos(wx * t),
            wy * ry * np.cos(wy * t),
            -wz * rz * np.sin(wz * t),
        ]
    )
    A = a**2 * np.array(
        [
            -(wx**2) * rx * np.sin(wx * t),
            -(wy**2) * ry * np.sin(wy * t),
            -(wz**2) * rz * np.cos(wz * t),
        ]
    )
    J = a**3 * np.array(
        [
            -(wx**3) * rx * np.cos(wx * t),
            -(wy**3) * ry * np.cos(wy * t),
            wz**3 * rz * np.sin(wz * t),
        ]
    )

    # return ca.horzcat(P,V,A,J)
    return np.hstack([P, V, A, J])

def coeff_to_pointsPoly(t, a, Tseq, XYZ_Coeff):
    n_all_poly = XYZ_Coeff.shape[0] // 3
    n_poly_perseg = n_all_poly // (Tseq.shape[0] - 1)
    X_Coeff = XYZ_Coeff[0:n_all_poly, 0]
    Y_Coeff = XYZ_Coeff[n_all_poly : 2 * n_all_poly, 0]
    Z_Coeff = XYZ_Coeff[2 * n_all_poly : 3 * n_all_poly, 0]

    def getCoeffCons(t):
        return np.array(
            [
                [1, 1 * t, 1 * t**2, 1 * t**3, 1 * t**4, 1 * t**5, 1 * t**6, 1 * t**7],
                [0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4, 6 * t**5, 7 * t**6],
                [0, 0, 2, 6 * t, 12 * t**2, 20 * t**3, 30 * t**4, 42 * t**5],
                [0, 0, 0, 6, 24 * t, 60 * t**2, 120 * t**3, 210 * t**4],
            ],
            dtype=np.float64,
        )

    if 0 < t <= Tseq[-1]:
        i = 0
        for k in range(0, Tseq.shape[0] - 1):
            if Tseq[k] < t <= Tseq[k + 1]:
                i = k

        Pxi = X_Coeff[i * n_poly_perseg : n_poly_perseg * (i + 1)]
        Pyi = Y_Coeff[i * n_poly_perseg : n_poly_perseg * (i + 1)]
        Pzi = Z_Coeff[i * n_poly_perseg : n_poly_perseg * (i + 1)]
        CoeffCons = getCoeffCons(t - Tseq[i])
        P = np.array(
            [
                np.dot(CoeffCons[0, :], Pxi),
                np.dot(CoeffCons[0, :], Pyi),
                np.dot(CoeffCons[0, :], Pzi),
            ]
        )
        V = a * np.array(
            [
                np.dot(CoeffCons[1, :], Pxi),
                np.dot(CoeffCons[1, :], Pyi),
                np.dot(CoeffCons[1, :], Pzi),
            ]
        )
        A = a**2 * np.array(
            [
                np.dot(CoeffCons[2, :], Pxi),
                np.dot(CoeffCons[2, :], Pyi),
                np.dot(CoeffCons[2, :], Pzi),
            ]
        )
        J = a**3 * np.array(
            [
                np.dot(CoeffCons[3, :], Pxi),
                np.dot(CoeffCons[3, :], Pyi),
                np.dot(CoeffCons[3, :], Pzi),
            ]
        )
    elif t > Tseq[-1]:
        Pxi = X_Coeff[-n_poly_perseg:]
        Pyi = Y_Coeff[-n_poly_perseg:]
        Pzi = Z_Coeff[-n_poly_perseg:]
        CoeffCons = getCoeffCons(Tseq[-1] - Tseq[-2])
        P = np.array(
            [
                np.dot(CoeffCons[0, :], Pxi),
                np.dot(CoeffCons[0, :], Pyi),
                np.dot(CoeffCons[0, :], Pzi),
            ]
        )
        V = np.array([0, 0, 0], dtype=np.float64)
        A = np.array([0, 0, 0], dtype=np.float64)
        J = np.array([0, 0, 0], dtype=np.float64)
    else:
        Pxi = X_Coeff[0:n_poly_perseg]
        Pyi = Y_Coeff[0:n_poly_perseg]
        Pzi = Z_Coeff[0:n_poly_perseg]
        CoeffCons = getCoeffCons(0)
        P = np.array(
            [
                np.dot(CoeffCons[0, :], Pxi),
                np.dot(CoeffCons[0, :], Pyi),
                np.dot(CoeffCons[0, :], Pzi),
            ]
        )
        V = np.array([0, 0, 0], dtype=np.float64)
        A = np.array([0, 0, 0], dtype=np.float64)
        J = np.array([0, 0, 0], dtype=np.float64)

    ref = np.hstack((P, V, A, J))
    return ref
