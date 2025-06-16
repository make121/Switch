# SMPL Motions Visualization

## Blender Vis

Import `npz` motion data into blender for visualization.

1. Download version `2.9.0` of blender and the [SMPL add-on](https://smpl.is.tue.mpg.de/index.html).

2. Add the SMPL object to the blender scene and run the `import
_motion_blender.py` script in the `scripting` bar to bind the motion data to the SMPL object.

## PyRender Vis

Run the following command to visualize motion through `PyRender`:

```
python smpl_render.py --filepath <PATH_TO_MOTION>
```

- `<PATH_TO_MOTION>` contains `.npz`