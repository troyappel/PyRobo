To run the simulation, run `main2.py`.

There are multiple configurable parameters. The number of bots is configurable at the top of `main.py`.
Simulation configuration is available in the RobotConfig class at the top of `robot.py`. To switch the point generators,
you can change the class used during initialization of all robots -- for example, change PointFilter to GPSLocalizer. 
You can change things marked with `CONFIG`.

This simulation requires matplotlib, numpy, and scipy.

