## PyRobo

This takes some direct inspiration from PythonRobotics (https://github.com/AtsushiSakai/PythonRobotics) and their example DWA control and markov localization.

To run the simulation, run `main2.py`.

There are multiple configurable parameters. The number of bots is configurable at the top of `main.py`.
Simulation configuration is available in the RobotConfig class at the top of `robot.py`. To switch the point generators,
you can change the class used during initialization of all robots -- for example, change PointFilter to GPSLocalizer. 
You can change things marked with `CONFIG`.

PyRobo splits the movement pipeline into several abstract classes.

MotionModel: How to represent a robot given input. Currently we support YawMotionModel, which simulates a robot with significant inertia and slow-to-change yaw and speed.

PointGenerator: How to get predicted robot states. Currently we have GPSLocater which generates gaussian predictions around a robot's true position, and ParticleFilter, which also uses distances to known landmarks.

Localizer: How to generate a 'footprint' for the robot based on points generated. Currently, we support COCALU as described here: https://link.springer.com/article/10.1007/s10514-018-9726-5

Cost: A cost to impose on potential robot trajectories. Multiple are supported.

Planner: How to choose and evaluate paths. Currently we use the Dynamic Window Approach (DWA) based on predicted robot state, choosing the lowest-cost trajectory. The planner also supports a state machine, such that when costs meet thresholds a new robot state is triggered with different weights. The example code has three states: "nospace", "default", and "goal".

This simulation requires matplotlib, numpy, and scipy.

