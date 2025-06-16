# SMPL Motion Retarget

Our code incorporates the retargeting pipelines from [MaskedMimic](https://github.com/NVlabs/ProtoMotions) and [PHC](https://github.com/ZhengyiLuo/PHC) - the former is built upon the differential inverse kinematics framework [Mink](https://github.com/kevinzakka/mink), while the latter employs gradient-based optimization.

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