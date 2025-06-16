
<br>
<p align="center">
<h1 align="center"><strong>KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills</strong></h1>
  <p align="center">
    <a href="https://shi-soul.github.io/" target="_blank">Weiji Xie</a><sup>* 1,2</sup>,
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=mJ7tn3MAAAAJ" target="_blank">Jinrui Han</a><sup>* 1,2</sup>,
    <a href="#" target="_blank">Jiakun Zheng</a><sup>* 1,3</sup>,
    <a href="https://scholar.google.com/citations?user=XhAo4J0AAAAJ&hl=zh-CN" target="_blank">Huanyu Li</a><sup>1,4</sup>,
    <a href="https://xinzheliu.github.io/" target="_blank">Xinzhe Liu</a><sup>1,5</sup>,
    <a href="https://scholar.google.com/citations?user=aVte5j4AAAAJ" target="_blank">Jiyuan Shi</a><sup>1</sup>,
    <a href="https://wnzhang.net" target="_blank">Weinan Zhang</a><sup>2</sup>,
    <a href="https://baichenjia.github.io/" target="_blank">Chenjia Bai</a><sup>&dagger;1</sup>,
    <a href="https://scholar.google.com.hk/citations?user=ahUibskAAAAJ" target="_blank">Xuelong Li</a><sup>1</sup>
    <br>
    * Equal Contribution  &dagger; Corresponding Author
    <br>
    <sup>1</sup>Institute of Artificial Intelligence (TeleAI), China Telecom  
    <sup>2</sup>Shanghai Jiao Tong University  
    <sup>3</sup>East China University of Science and Technology  
    <sup>4</sup>Harbin Institute of Technology  
    <sup>5</sup>ShanghaiTech University
  </p>
</p>

<div id="top" align="center">

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2502.13013-orange)](https://arxiv.org/abs/2502.13013) -->
[![](https://img.shields.io/badge/Project-%F0%9F%9A%80-pink)](https://kungfu-bot.github.io/)

</div>

## Demo

[![demo](demo.jpg "demo")]()


## News
- \[2025-06\] We release the code and paper for PBHC.

## Contents

- [Demo](#demo)
- [News](#news)
- [Contents](#contents)
- [About](#about)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## About



[![overview](overview.jpg "overview")]()


This is the official implementation of the paper [KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills](https://kungfu-bot.github.io/).


Our paper introduces a physics-based control framework that enables humanoid robots to learn and reproduce challenging motions through multi-stage motion processing and adaptive policy training.

This repository includes:
- Motion processing pipeline:
  - Collect human motion from multiple datasources (video, LAFAN, AMASS, etc.) to a unified SMPL format (`motion_source/`)
  - Filter, correct and retarget human motion to the robot (`smpl_retarget/`)
  - Visualize and analyze the processed motion (`smpl_vis/`, `robot_motion_process/`)
- RL-based motion imitation framework (`humanoidverse/`):
  - Simulation training in IsaacGym
  - Sim-to-sim deployment in MuJoCo
- Example data(`example/`)
  - a pieces of motion data in our experiments(`example/motion_data/`)
  - a pretrained policy checkpoint (`example/pretrained_hors_stance_pose/`)



## Usage

- Refer to `INSTALL.md` for environment setup and installation instructions.

- Each module folder (e.g., `humanoidverse`, `smpl_retarget`) contains a dedicated `README.md` explaining its purpose and usage.

## Folder Structure

- `description`: provide description file for SMPL and G1 robot.
- `motion_source`: docs for getting SMPL format data.
- `smpl_retarget`: tools for SMPL to G1 robot retargeting.
- `smpl_vis`: tools for visualizing SMPL format data.
- `robot_motion_process`: tools for processing robot format motion. Including visualization, interpolation, and trajectory analysis.
- `humanoidverse`: training RL policy
- `example`: example motion and ckpt for using PBHC


## Citation

If you find our work helpful, please cite:

```bibtex
@article{xie2025kungfubot,
  title={KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills},
  author={Xie, Weiji and Han, Jinrui and Zheng, Jiakun and Li, Huanyu and Liu, Xinzhe and Shi, Jiyuan and Zhang, Weinan and Bai, Chenjia and Li, Xuelong},
  journal={arXiv preprint},
  year={2025}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [ASAP](https://github.com/LeCAR-Lab/ASAP): We use `ASAP` library to build our RL codebase.
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl): We use `rsl_rl` library to train the RL policies.
- [Unitree](https://www.unitree.com/): We use `Unitree G1` as our testbed.
- [Maskedmimic](https://github.com/NVlabs/ProtoMotions): We use the retargeting pipeline in `Maskedmimic`, which based on [Mink](https://github.com/kevinzakka/mink).
- [PHC](https://github.com/ZhengyiLuo/PHC): We incorporate the retargeting pipeline from `PHC` into our implementation.
- [GVHMR](https://github.com/zju3dv/GVHMR): We use `GVHMR` to extract motions from videos.