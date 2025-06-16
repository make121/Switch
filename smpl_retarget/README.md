# SMPL Motion Retarget

## Mink Retarget

First install `poselib`:
```
cd poselib
pip install -e .
```

Retarget command:
```
python mink_retarget/convert_fit_motion.py <PATH_TO_MOTION_FOLDER>
```

`<PATH_TO_MOTION_FOLDER>` is the root folder of motion data. The motion data folder should be like this:

```
motion_data/
├── video_motion/
|    └── video1.npz
|    └── video2.npz
|    └── video3.npz
|    └── ...
└── amass/
     └── reverse_spin.npz
     └── wushu_form.npz
     └── ...
```
In above case, `<PATH_TO_MOTION_FOLDER>` is `motion_data/`

## PHC Retarget

Download the [SMPL](https://smpl.is.tue.mpg.de/) v1.1.0 parameters and place them in the `smpl_model/smpl/` folder. Rename the files basicmodel_neutral_lbs_10_207_0_v1.1.0, basicmodel_m_lbs_10_207_0_v1.1.0.pkl, basicmodel_f_lbs_10_207_0_v1.1.0.pkl to SMPL_NEUTRAL.pkl, SMPL_MALE.pkl and SMPL_FEMALE.pkl.

The folder structure should be like this:
```
smpl_model/smpl/
├── SMPL_FEMALE.pkl
├── SMPL_MALE.pkl
└── SMPL_NEUTRAL.pkl
```

Retarget command:
```
python phc_retarget/fit_smpl_motion.py robot=unitree_g1_29dof_anneal_23dof +motion=<PATH_TO_MOTION_FOLDER>
```