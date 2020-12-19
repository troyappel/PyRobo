import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import random

class MotionModel(object):
    def __init__(self, x, config, u):
        self.x = x
        self.config = config
        self.x_est = x
        self.u = u

    def model(self, u, x=None):
        pass

    def step(self, u):
        self.x = self.model(u)

class PointGenerator(object):
    def __init__(self, motion_model):
        pass

    def get_points(self, u, rf_id):
        pass

class Localizer(object):
    def __init__(self, config):
        self.config = config

    def get_footprint(self, px, pw):
        pass

    def bound_points(self, footprint, resolution=4):
        pass

class Cost(object):
    """
    Cost contains a cost function, and also acts as a FSM controller; if the cost function is within
    some range, we proceed the FSM
    """
    def __init__(self, mappings, weight_mappings):
        self.mappings = mappings
        self.weight_mappings = weight_mappings

    def triggered(self, val, cur_state):
        """
        :param val:
        :param cur_state:
        :return: New mapping if triggered, None o/w
        """

        for _min, _max, this_state, next_state in self.mappings:
            if _min is not None and _min <= val <= _max and cur_state == this_state:
                print(this_state, cur_state, val)
                return next_state

        return None

    def weight(self, state):
        if state in self.weight_mappings.keys():
            return self.weight_mappings[state]
        else:
            return 0

    # Returns number representing the cost of that state
    def cost(self, config, **kwargs):
        pass

    def visualize(self, plt, ax, config, **kwargs):
        pass

class Planner(object):
    def __init__(self, config, motion_model, costs, state="default"):
        self.config = config
        self.motion_model = motion_model
        self.costs = costs

        self.state = state

    def control(self, x, goal, ob, footprint):
        pass

    def get_costs(self, **kwargs):
        cost_val_weights = [(cost.cost(self.config, **kwargs), cost.weight(self.state)) for cost in self.costs]

        return sum([x * y for x, y in cost_val_weights]), cost_val_weights

    def run_fsm(self, cost_val_weights):
        if cost_val_weights is None:
            return

        for i in reversed(range(0,len(self.costs))):
            cost = self.costs[i]
            cost_val, weight = cost_val_weights[i]

            new_state = cost.triggered(cost_val, self.state)

            if new_state is not None and new_state != self.state:
                self.state = cost.triggered(cost_val, self.state)
                break

class Robot(object):
    def __init__(self, point_generator, localizer, controller, motion_model, goal, priority):
        self.point_generator = point_generator
        self.localizer = localizer
        self.controller = controller
        self.motion_model = motion_model
        self.px = []
        self.pw = []
        self.footprint = None
        self.goal = goal
        self.trajectory = []

        self.priority = priority

    def step(self):
        self.motion_model.step()
