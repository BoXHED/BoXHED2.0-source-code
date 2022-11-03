# BoXHED2.0

What’s new (over BoXHED 1.0):
 - Allows for survival data far beyond right censoring (specifically, Aalen’s multiplicative intensity setting). Examples include left truncation and recurrent events.
 - Significant speed improvement
 - Multicore CPU and GPU support

Please refer to [BoXHED2.0 paper](https://arxiv.org/abs/2103.12591) for details, which builds on [BoXHED1.0 paper](http://proceedings.mlr.press/v119/wang20o/wang20o.pdf) (ICML 2020). The theoretical underpinnings for BoXHED is provided [here](https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-4/Boosted-nonparametric-hazards-with-time-dependent-covariates/10.1214/20-AOS2028.full) (Annals of Statistics 2021).

## Prerequisites
The software developed and tested in Linux and Mac OS environments. The requirements are the following:
- cmake  (>=3.18.2) (CMake is part of Visual Studio installation for Windows)
- Python (=3.8)
- CUDA   (=11.1) (only if GPU support is needed)
- conda

For Windows users, using [Git for Windows](https://gitforwindows.org/) would significantly ease the installation process. For unifying the approach, we refer to the command line interface as 'terminal' but it could be Git for Windows for Windows users. Windows users need to have Visual Studio 17 2022 toolset installed.

When installing Visual Studio 17 2022 toolset, among "Workloads" select "Desktop Development with C++" in the "Desktop and Mobile" section. A selection menu shows up on the right where you need to make the following selections:
![vs_windows_selection](https://user-images.githubusercontent.com/34462617/198723876-38a85c80-4e50-4fe7-8a8a-1ac3c020d346.jpg)

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

Clone this repository:
```
git clone https://github.com/BoXHED/BoXHED2.0.git
```
Then, go to the directory:
```
cd BoXHED2.0
```

## Installing BoXHED2.0 (CPU training only)
For CPU training only, you can install BoXHED2.0 from pip:
```
pip install boxhed
```

Now install Jupyter Notebook:
```
pip install jupyter
```

Run the *BoXHED2_tutorial.ipynb* file for a quick demonstration of how to train/test a BoXHED model on a synthetic dataset. Please refer to this file for proper usage of BoXHED2.0.
```
jupyter notebook BoXHED2_tutorial.ipynb
``` 

## Installing BoXHED 2.0 (with GPU support)

For adding GPU support, BoXHED2.0 needs to be build from source. 

Now install numpy, pandas, scikit-learn, pytz, py3nvml, matplotlib, and CudaToolkit by:
```
source conda_install_packages.sh
```

Note that you need CUDA 11.1 installed. In case you do not have it already, you can download CUDA 11.1 from [here](https://developer.nvidia.com/cuda-11.1.1-download-archive).

Subsequently you can install BoXHED2.0 by running:
```
./setup.sh
```

The installer logs its activity in the terminal. If it states that the installation has been unsuccessful, you may check the *setup_log.txt* file created within the current directory for a more detailed description of what has gone wrong.  

According to our experience, one of the most prevalent problems is that CMake cannot find CUDA the compiler or needs some arguments set depending on your machine. If that is the case, please see the instructions at line 65 of the file *setup.sh* to properly set the missing arguments.

Run the *BoXHED2_tutorial.ipynb* file for a quick demonstration of how to train/test a BoXHED model on a synthetic dataset. Please refer to this file for proper usage of BoXHED2.0.
```
jupyter notebook BoXHED2_tutorial.ipynb
``` 