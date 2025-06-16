
# Robot Motion Processing

In this folder, we provide tools for visualizing and analyzing the processed motion and the motion trajectory from simulation.

All of the retargeted reference motion and motion trajectory collected from simulation take the same data format. `motion_readpkl.py` provide an example code to load them.

To collect the motion trajectory in IsaacGym, use `humanoidverse/sample_eps.py`. For MuJoCo, use `humanoidverse/urci.py`, the program will automatically save the motion trajectory to a pickle file when each episode ends.



## Visualization

`vis_q_mj.py` and `vis_rr.py` are both used to visualize the motion data. These two scripts load the same data format, but have a different GUI.

Usage:

```bash
python robot_motion_process/vis_q_mj.py +motion_file=...
```

```bash
python robot_motion_process/vis_rr.py --filepath=...
```





## Interpolation

`motion_interpolation_pkl.py` is used to preprocess the motion data. Given a default robot pose, this script will add a linear interpolation between the default pose and the motion frames at both the beginning and the end. So that the processed motion data all begin and end with the default pose. 

This is useful for deployment since it's inconvenient to initialize the real-world robot to the beginning pose of each motion. Some pose may be strange so that the robot cannot stand stably.


## Trajectory Analysis



## Manually Label Contact Mask