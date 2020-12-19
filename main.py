import robot as rb
import abstractclasses as ab
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib
import numpy as np
import math

# CONFIG: CHANGE NUMBER OF BOTS IN CIRCLE
N_BOTS = 6

# CONFIG: CHANGE POSE
POSE = "circle"

# CONFIG: RANDOMIZE ROBOT SPECS
randomize_config = False

def start_and_goal(i, n_bots, pose):

    if pose == "circle":
        angle = math.pi * 2 * i / n_bots

        x = np.zeros((5, 1))
        dest = np.zeros((2,1))

        x[0] = 5 + 5*math.cos(angle)
        x[1] = 5 + 5*math.sin(angle)
        x[2] = angle + math.pi
        dest[0] = 10 - x[0]
        dest[1] = 10 - x[1]

        return x, dest

    elif pose == "linear":
        x = np.zeros((5, 1))
        dest = np.zeros((2,1))

        x[0] = 0
        x[1] = 10 * (i / n_bots)
        x[2] = 0

        dest[0] = 10
        dest[1] = 10 * (i / n_bots)

        return x, dest

    elif pose == "random":
        x = np.zeros((5, 1))
        dest = np.zeros((2,1))

        x[0] = np.random.random() * 10
        x[1] = np.random.random() * 10
        x[2] = np.random.random() * math.pi * 2

        angle = math.pi * 2 * i / n_bots

        dest[0] = 5 - 5*math.cos(angle)
        dest[1] = 5 - 5*math.sin(angle)


        return x, dest


def main():
    print(__file__ + " start!!")

    bots = []

    fig, ax = plt.subplots()

    # RF_ID positions [x, y]
    rf_id = np.array([[10.0, 0.0]])

    prev_trajs = []

    # CONFIG: ADD POINTS AS LISTS LIKE [1,2],[2,3]
    humans = []
    static_obs = []

    for i in range(0, N_BOTS):
        start, goal = start_and_goal(i, N_BOTS, POSE)

        u = np.array([[0, 0.0]]).T

        priority = 20 + i * 5

        # CONFIG: CHANGE WEIGHTS OR REMOVE COSTS
        costs = [
            rb.ObCost({(0, 0.7, 'nospace', 'default'), (1, float('inf'), 'default', 'nospace')}, {'default': priority, 'nospace': priority + 20, 'goal': priority}),
            rb.GoalCost({(0, 2, 'default', 'goal')}, {'default': 15, 'nospace': 5, 'goal': 100}),
            rb.HighwayCost({}, {'default': 3, 'nospace': 1}, 5, 3, 7, 3),
            rb.SpeedCost({}, {'default': 3, 'nospace': 1, 'goal': 0}),
            rb.YawCost({}, {'default': 1, 'nospace': 1})
        ]

        config = rb.RobotConfig()

        if randomize_config:
            config.max_speed += np.random.normal(0, 0.005, 1)[0]
            config.min_speed += np.random.normal(0, 0.0002, 1)[0]
            config.max_yaw_rate += np.random.normal(0, 0.01, 1)[0]

            config.max_accel += np.random.normal(0, 0.01, 1)[0]

            config.GPS_stdev += np.random.normal(0, 0.0005, (5, 1))

            config.Q[0, 0] += np.random.normal(0, 0.005, 1)  # range error
            config.R[0, 0] += abs(np.random.normal(0, 0.0005, 1))  # inp error
            config.R[1, 1] += abs(np.random.normal(0, 0.0005, 1))  # inp error


        model = rb.YawMotionModel(start, config, u)
        particles = rb.ParticleFilter(config, model, start)
        localizer = rb.COCALU(config)
        dwa_control = rb.DWA(config, model, costs)

        bots.append(ab.Robot(particles, localizer, dwa_control, model, goal, 1))

    for i in range(0, 10**10):
        # Step all bots
        plt.cla()
        for bot in bots:
            bot.motion_model.step(bot.motion_model.u)
            bot.px, bot.pw = bot.point_generator.get_points(bot.motion_model.u, rf_id)
            bot.motion_model.x_est = bot.px.dot(bot.pw.T)

            bot.footprint = bot.localizer.get_footprint(bot.px[:2].T, bot.pw.T)

        traj_lst = []

        for human in humans:
            human[0] += np.random.random(1) / 5 - 0.1
            human[1] += np.random.random(1) / 5 - 0.1

        # Generate next input
        for bot in bots:
            if bot.footprint is None:
                continue

            other_bounds = []
            for _bot in bots:
                if _bot == bot:
                    continue

                _trajectory, _footprints = _bot.controller.predict_trajectory(_bot.motion_model.x_est.flatten(), _bot.motion_model.u[0][0],
                                                _bot.motion_model.u[1][0], None)

                other_bounds.append(_trajectory)

            ctrl, traj = bot.controller.control(bot.motion_model.x_est.flatten(), np.array(bot.goal), np.array(other_bounds),
                                                      bot.footprint, humans, static_obs, i)
            bot.motion_model.u[0] = ctrl[0]
            bot.motion_model.u[1] = ctrl[1]

            bot.trajectory = traj

            traj_lst.append(bot.motion_model.x.flatten())

        prev_trajs.append(traj_lst)


        # Print chart
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        plt.plot(rf_id[:, 0], rf_id[:, 1], "*k")

        patches = []

        for bot in bots:
            plt.plot(bot.px[0, :], bot.px[1, :], ".b")
            plt.plot(bot.motion_model.x[0], bot.motion_model.x[1], 'ro')
            plt.quiver(bot.motion_model.x[0], bot.motion_model.x[1], math.cos(bot.motion_model.x[2]), math.sin(bot.motion_model.x[2]))

            plt.plot(bot.goal[0], bot.goal[1], 'go')

            plt.plot(bot.trajectory[:, 0], bot.trajectory[:, 1], "-r")

            if bot.footprint is None:
                continue

            patches.append(Polygon(bot.footprint.points[bot.footprint.vertices][:, :2], True))

        for i in range(N_BOTS):
            plt.plot(np.array(prev_trajs)[:, i, 0], np.array(prev_trajs)[:, i, 1], "-m")

        for human in humans:
            plt.plot(human[0], human[1], ".m")

        for static_ob in static_obs:
            plt.plot(static_ob[0], static_ob[1], "ok")

        for cost in costs:
           cost.visualize(plt, ax, None)

        p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.2)


        colors = 100*np.array([i / N_BOTS for i in range(0, N_BOTS)])
        p.set_array(np.array(colors))
        
        ax.add_collection(p)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)


if __name__ == '__main__':
    main()
