# BoXHED2.0

What’s new (over BoXHED 1.0):
 - Allows for survival data far beyond right censoring (specifically, Aalen’s multiplicative intensity setting). Examples include left truncation and recurrent events.
 - Significant speed improvement
 - Multicore CPU and GPU support

Please refer to [BoXHED2.0 Paper](https://arxiv.org/abs/2103.12591) for details, which builds on [BoXHED1.0 Paper](http://proceedings.mlr.press/v119/wang20o/wang20o.pdf) (ICML 2020).

## Prerequisites
The software developed and tested in Linux and Mac OS environments. The requirements are the following:
- cmake  (>=3.18.2)
- Python (>=3.8)
- conda

For Windows users, using [Git for Windows](https://gitforwindows.org/) would significantly ease the installation process. For unifying the approach, we refer to the command line interface as 'terminal' but it could be Git for Windows for Windows users.

We highly recommend devoting a conda environment to BoXHED 2.0. This step makes sure BoXHED 2.0 will not interfere with XGBoost (the library we have borrowed from extensively) when installed. This implementation uses python 3.8.
For installing the conda environment please open a terminal and do the following:

First create the conda environment:
```
conda create -n boxhed2.0 python=3.8
```

then activate it
```
conda activate boxhed2.0
```

Please make sure to clone this repository:
```
git clone https://github.com/BoXHED/BoXHED2.0.git
```
Then, go to the directory:
```
cd BoXHED2.0
```
now install numpy, pandas, scikit-learn, pytz, py3nvml and matplotlib by:
```
source conda_install_packages.sh
```

Subsequently you can install BoXHED2.0 by running:
```
source setup.sh
```
Here are the flags that can be passed to the installer:
- '-v': If you are a Windows user and are installing for Windows Visual Studio, the version can be passed using this flag (14, 15, and 16 are supported). For 14 for example, you may run 'bash setup.sh -v 14'
- '-g': If you want the code to be compiled with GPU support, you may pass the '-g' flag. Please note that this is only supported for Linux users at the moment. 

then run the *main.py* file for a quick demonstration of how to train/test a BoXHED model on a synthetic dataset. Please refer to this file for proper usage of BoXHED2.0.
```
python main.py
``` 

Please note that everytime you relocate the code, you need to run bash setup.sh again.
