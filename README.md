# iDFusion: Globally Consistent Dense 3D Reconstruction from RGB-D and Inertial Measurements.

## Introduction
This is the official code repository for iDFusion, which integrates the global optimization of IMU information into [FlashFusion](https://github.com/THU-luvision/Flashfusion).

This is a project from LuVision SIGMA, Tsinghua University. Visit our website for more interesting works: http://www.luvision.net/

## License
This project is released under the [GPLv3 license](LICENSE). We only allow free use for academic use. For commercial use, please contact us to negotiate a different license by: `fanglu at tsinghua.edu.cn`

## Citing

If you find our code useful, please kindly cite our paper:

```bibtex
@inproceedings{zhong2019idfusion,
  title={iDFusion: Globally Consistent Dense 3D Reconstruction from RGB-D and Inertial Measurements},
  author={Zhong, Dawei and Han, Lei and Fang, Lu},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={962--970},
  year={2019}
}
@inproceedings{han2018flashfusion,
  title={FlashFusion: Real-time Globally Consistent Dense 3D Reconstruction using CPU Computing.},
  author={Han, Lei and Fang, Lu},
  booktitle={Robotics: Science and Systems},
  volume={1},
  number={6},
  pages={7},
  year={2018}
}

```

## Environment setup

### Preliminary Requirements:
* Ubuntu 16.04/18.04
* Intel cpu 
* ROS Kinetic (follow the instructions on http://wiki.ros.org/kinetic)

### Compile

```bash
source prepare.sh
mkdir build
cd build
cmake ..
make -j
```

### Usage 
```
1. Start ros:     ./roscore
2. Run iDFusion:  ./af
3. Publish data:  ./rosbag play 0.bag
```
