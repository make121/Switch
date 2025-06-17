# SMPL Motions Visualization

This folder describes how to visualize the SMPL format motion data. We provide two methods: `Blender` and `PyRender`.

## Blender Vis

[Blender](https://www.blender.org/) is an open source software for 3D CG. It's also a powerful tool for motion visualization.

Import `npz` motion data into blender for visualization.

1. Download version `2.9.0` of blender and the [SMPL add-on](https://smpl.is.tue.mpg.de/index.html).

2. Add the SMPL object to the blender scene and run the `import_motion_blender.py` script in the `scripting` bar to bind the motion data to the SMPL object.

## PyRender Vis

Run the following command to visualize motion through `PyRender`:

```
python smpl_render.py --filepath <PATH_TO_MOTION>
```

- `<PATH_TO_MOTION>` contains `.npz`