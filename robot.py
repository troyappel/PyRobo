import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import random

import abstractclasses as ab

# CONFIG: change these for effects
class RobotConfig:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.05  # [m/s]
        self.max_yaw_rate = 120.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 2  # [m/ss]
        self.max_delta_yaw_rate = 240.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.02  # [m/s]
        self.yaw_rate_resolution = 2 * math.pi / 180.0  # [rad/s]
        self.dt = 0.2  # [s] Time tick for motion prediction
        self.predict_time = 7  # [s]
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked

        self.N_dwa_samples = 150

        self.obstacle_dist = 7

        self.yaw_cost_gain = 0.01

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1  # [m] for collision check

        self.GPS_stdev = np.array([[0.3], [0.3], [0.01], [0.01], [0.01]])

        self.human_cost_gain = 2


        # Used for pointgenerator
        self.MAX_RANGE = 20
        # Estimation parameter of PF
        self.Q = np.diag([0.4]) ** 2  # range error
        self.R = np.diag([0.03, np.deg2rad(2)]) ** 2  # input error
        #  Simulation parameter
        self.Q_sim = np.diag([0.4]) ** 2
        self.R_sim = np.diag([0.03, np.deg2rad(2)]) ** 2
        self.MAX_RANGE = 20.0  # maximum observation range
        # Particle filter parameter
        self.NP = 50  # Number of Particle
        self.NTh = self.NP / 2.0  # Number of particle for re-sampling

        self.C_epsilon = 0.05

class YawMotionModel(ab.MotionModel):
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    def __init__(self, x, config, u):
        super(YawMotionModel, self).__init__(x, config, u)

    def model(self, u, x=None):
        if x is None:
            x = self.x

        # # Assume straight-line motion

        x[2] += u[1] * self.config.dt
        x[0] += u[0] * math.cos(x[2]) * self.config.dt
        x[1] += u[0] * math.sin(x[2]) * self.config.dt
        x[3] = u[0]
        x[4] = u[1]

        return x

    def step(self, u):
        self.x = self.model(u)


class GPSLocater(ab.PointGenerator):
    def __init__(self, config, motion_model, x_init):
        super().__init__(self)
        self.config = config
        self.motion_model = motion_model

    def get_points(self, u, rf_id):
        px = np.random.normal(
            loc=self.motion_model.x[:, 0],
            scale=self.config.GPS_stdev[:, 0],
            size=(self.config.NP, 5)
        )
        pw = np.ones((self.config.NP, 1)) * 1 / self.config.NP
        return px.T, pw.T

class ParticleFilter(ab.PointGenerator):
    def __init__(self, config, motion_model, x_init):
        super().__init__(self)
        self.config = config
        self.motion_model = motion_model

        self.px = np.tile(x_init, self.config.NP)
        self.pw = np.zeros((1, self.config.NP)) + 1.0 / self.config.NP  # Particle weight


    def observation(self, u, rf_id):
        x_true = self.motion_model.x

        # add noise to gps x-y
        z = np.zeros((0, 3))

        for i in range(len(rf_id[:, 0])):
            dx = x_true[0, 0] - rf_id[i, 0]
            dy = x_true[1, 0] - rf_id[i, 1]
            d = math.hypot(dx, dy)
            if d <= self.config.MAX_RANGE:
                dn = d + np.random.randn() * self.config.Q_sim[0, 0] ** 0.5  # add noise
                zi = np.array([[dn, rf_id[i, 0], rf_id[i, 1]]])
                z = np.vstack((z, zi))

        # add noise to input
        ud1 = u[0, 0] + np.random.randn() * self.config.R_sim[0, 0] ** 0.5
        ud2 = u[1, 0] + np.random.randn() * self.config.R_sim[1, 1] ** 0.5
        ud = np.array([[ud1, ud2]]).T

        return z, ud

    def gauss_likelihood(self, x, sigma):
        p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
            math.exp(-x ** 2 / (2 * sigma ** 2))

        return p

    def pf_localization(self, px, pw, z, u):
        """
        Localization with Particle filter -- now returns just weighted points
        """
        for ip in range(self.config.NP):
            x = np.array([px[:, ip]]).T
            w = self.pw[0, ip]

            #  Predict with random input sampling
            ud1 = u[0, 0] + np.random.randn() * self.config.R[0, 0] ** 0.5
            ud2 = u[1, 0] + np.random.randn() * self.config.R[1, 1] ** 0.5
            ud = np.array([[ud1, ud2]]).T
            x = self.motion_model.model(ud, x)

            #  Calc Importance Weight
            for i in range(len(z[:, 0])):
                dx = x[0, 0] - z[i, 1]
                dy = x[1, 0] - z[i, 2]
                pre_z = math.hypot(dx, dy)
                dz = pre_z - z[i, 0]
                w = w * self.gauss_likelihood(dz, math.sqrt(self.config.Q[0, 0]))

            px[:, ip] = x[:, 0]
            pw[0, ip] = w

        pw = pw / pw.sum()  # normalize

        N_eff = 1.0 / (pw.dot(pw.T))[0, 0]  # Effective particle number
        if N_eff < self.config.NTh:
            px, pw = self.re_sampling(px, pw)

        return px, pw

    def re_sampling(self, px, pw):
        """
        low variance re-sampling
        """
        w_cum = np.cumsum(pw)
        base = np.arange(0.0, 1.0, 1 / self.config.NP)
        re_sample_id = base + np.random.uniform(0, 1 / self.config.NP)
        indexes = []
        ind = 0
        for ip in range(self.config.NP):
            while re_sample_id[ip] > w_cum[ind]:
                ind += 1
            indexes.append(ind)

        px = px[:, indexes]
        pw = np.zeros((1, self.config.NP)) + 1.0 / self.config.NP  # init weight

        return px, pw

    def get_points(self, u, rf_id):
        z, ud = self.observation(u, rf_id)
        px, pw = self.pf_localization(self.px, self.pw, z, ud)

        self.px = px
        self.pw = pw

        return px, pw


class COCALU(ab.Localizer):
    def __init__(self, config):
        super(COCALU, self).__init__(config)

    def get_footprint(self, px, pw):
        """
        :param px: points
        :param pw: point weights
        :return: scipy.spatial.ConvexHull object of the footprint
        """

        cur_points = np.array(px, copy=True)
        cur_weights = np.array(pw, copy=True)

        bound = 0
        hull = None

        while bound < self.config.C_epsilon:
            try:
                hull = ConvexHull(cur_points)
            except Exception:
                cur_points = cur_points + np.random.rand(*cur_points.shape)
                continue

            # hull.vertices is indices into array
            hull_weight = pw[hull.vertices].sum()

            # Increment bound
            bound += hull_weight

            # Remove convex hull from working set
            cur_points = np.delete(cur_points, hull.vertices, axis=0)
            cur_weights = np.delete(cur_weights, hull.vertices, axis=0)

        return hull

    def bound_points(self, footprint, resolution=4):
        N = len(footprint.vertices)

        bound_points = None

        for i in range(0, N):
            btw = np.linspace(footprint.points[i], footprint.points[(i + 1) % N], num=resolution)

            if bound_points is None:
                bound_points = btw
            else:
                bound_points = np.concatenate((bound_points, btw))

        return bound_points

class GoalCost(ab.Cost):
    def cost(self, config, **kwargs):
        """
            calc to goal cost with angle difference
        """
        goal = kwargs["goal"]
        trajectory = kwargs["trajectory"]

        # Break ties in favor of facing goal
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle))) / 10

        # Use crow distance, since just checking heading creates loops
        cost += math.sqrt(dx[0] ** 2 + dy[0] ** 2)

        return cost

class ObCost(ab.Cost):
    def cost(self, config, **kwargs):
        trajectory = kwargs["trajectory"]
        ob = kwargs["ob"]
        footprints = kwargs["footprints"]
        humans = kwargs["humans"]
        static_obs = kwargs["static_obs"]

        N = len(footprints[0])

        n_obs = len(ob)

        min_r = float("inf")

        if len(ob) > 0:

            tiled_ob_x = np.tile(ob[:, :, 0][:, :, np.newaxis], (1, 1, N))
            tiled_ob_y = np.tile(ob[:, :, 1][:, :, np.newaxis], (1, 1, N))

            tiled_fp_x = np.tile(footprints[:, :, 0], (n_obs, 1, 1))
            tiled_fp_y = np.tile(footprints[:, :, 1], (n_obs, 1, 1))

            d_x_v = tiled_ob_x - tiled_fp_x
            d_y_v = tiled_ob_y - tiled_fp_y

            r_v = np.hypot(d_x_v, d_y_v)

            if np.array(r_v <= config.robot_radius).any():
                return float("inf")

            min_r = min(min_r, np.amin(r_v))

        # Calc if next to humans
        for human in humans:
            d_x_h = footprints[:, :, 0] - human[0]
            d_y_h = footprints[:, :, 1] - human[1]

            r_h = np.hypot(d_x_h, d_y_h) / config.human_cost_gain

            if np.array(r_h <= config.robot_radius).any():
                return float("Inf")

            min_r = min(min_r, np.amin(r_h))

        for static_ob in static_obs:
            d_x_h = footprints[:, :, 0] - static_ob[0]
            d_y_h = footprints[:, :, 1] - static_ob[1]

            r_h = np.hypot(d_x_h, d_y_h)

            if np.array(r_h <= config.robot_radius).any():
                return float("Inf")

            min_r = min(min_r, np.amin(r_h))


        return 1.0 / (min_r - config.robot_radius) ** 2  # OK



class HighwayCost(ab.Cost):
    def __init__(self, mappings, weight_mappings, weight, min_y, max_y, speed_lim, vertical=True):
        super(HighwayCost, self).__init__(mappings, weight_mappings)
        self.min_y = min_y
        self.max_y = max_y
        self.speed_lim = speed_lim

        self.vertical = vertical

    def cost(self, config, **kwargs):
        trajectory = kwargs["trajectory"]
        footprints = kwargs["footprints"]

        footprint_idx = 1 if self.vertical else 0

        footprints_desired = footprints[:, :, footprint_idx]

        N_POINTS = len(footprints[0])

        bad_points = np.logical_and(self.min_y < footprints_desired, footprints_desired < self.max_y)

        pos_cost = np.count_nonzero(bad_points) / N_POINTS

        vel_cost = sum([1 if p[3] > self.speed_lim else 0 for p in trajectory]) / (config.predict_time * config.dt) \
            if bad_points.any() else 0

        return (pos_cost + vel_cost) / 2

    def visualize(self, plt, ax, config, **kwargs):
        # color = 'red' if self.weight > 0 else 'blue'
        if self.vertical:
            ax.axhspan(self.min_y, self.max_y, color='red', alpha=0.2)
        else:
            ax.axvspan(self.min_y, self.max_y, color='red', alpha=0.2)

class SpeedCost(ab.Cost):
    def cost(self, config, **kwargs):
        trajectory = kwargs["trajectory"]
        return (config.max_speed - trajectory[-1, 3])**2

class YawCost(ab.Cost):
    def cost(self, config, **kwargs):
        return kwargs["yaw_rate"] ** 2


class DWA(ab.Planner):
    def __init__(self, config, motion_model, costs):
        super(DWA, self).__init__(config, motion_model, costs)

    def control(self, x, goal, ob, footprint, humans, static_obs, timestep):
        """
        Dynamic Window Approach control
        """
        dw = self.calc_dynamic_window(x)

        u, trajectory = self.calc_control_and_trajectory(x, dw, goal, ob, footprint, humans, static_obs, timestep)

        return u, trajectory

    def calc_dynamic_window(self, x):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [self.config.min_speed, self.config.max_speed,
              -self.config.max_yaw_rate, self.config.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [x[3] - self.config.max_accel * self.config.dt,
              x[3] + self.config.max_accel * self.config.dt,
              x[4] - self.config.max_delta_yaw_rate * self.config.dt,
              x[4] + self.config.max_delta_yaw_rate * self.config.dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def predict_trajectory(self, x_init, v, y, footprint_vertices):

        """
        predict trajectory with an input
        return tuple (trajectory, predicted_robot_bounds)
        """

        x = np.array(x_init)
        trajectory = np.array(x)

        footprints = None

        if footprint_vertices is not None:
            fp = np.array([footprint_vertices[:, 0] - x[0], footprint_vertices[:, 1] - x[1]]).T[:, :, np.newaxis]
            footprints = np.tile(np.array([footprint_vertices]), (1, 1, 1))

            rot_mat = np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])

        time = 0
        while time <= self.config.predict_time:
            x = self.motion_model.model([v, y], x=x)
            trajectory = np.vstack((trajectory, x))
            time += self.config.dt


            if footprint_vertices is not None:
                # Rotate about centroid
                # print(y)
                fp = rot_mat @ fp

                fp_trans = np.array([fp[:, 0] + x[0], fp[:, 1] + x[1]]).T
                footprints = np.vstack((footprints, fp_trans))

        return trajectory, footprints

    def calc_control_and_trajectory(self, x, dw, goal, ob, footprint, humans, static_obs, timestep):
        """
        calculation final input with dynamic window
        """
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])
        best_costs = None

        # evaluate all trajectory with sampled input in dynamic window
        # Choose samples
        for _ in range(self.config.N_dwa_samples):
            v = random.uniform(dw[0], dw[1])
            y = random.uniform(dw[2], dw[3])

            assert(isinstance(dw[2], float))
            assert(isinstance(dw[3], float))
            assert(isinstance(y, float))
            trajectory, footprints = self.predict_trajectory(x_init, v, y, footprint.points[footprint.vertices])

            # calc cost
            kwargs = {
                "trajectory": trajectory,
                "goal": goal,
                "ob": ob,
                "footprints": footprints,
                "humans": humans,
                "static_obs": static_obs,
                "time": timestep,
                "yaw_rate": y
            }

            final_cost, cost_weights = self.get_costs(**kwargs)

            if final_cost < 100000:
                plt.plot(trajectory[:, 0], trajectory[:, 1], "-b", alpha=0.1)

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                best_costs = cost_weights

                if abs(best_u[0]) < self.config.robot_stuck_flag_cons \
                        and abs(x[3]) < self.config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -self.config.max_delta_yaw_rate

        if min_cost == float("inf"):
            best_u[0] = -0.2
            best_u[1] = 0

        self.run_fsm(best_costs)


        return best_u, best_trajectory

