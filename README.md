# BoXHED2.0

What’s new (over BoXHED1.0):
 - Allows for survival data beyond right censoring, including recurrent events
 - Significant speed improvement
 - Multicore CPU and GPU support


Please refer to [BoXHED2.0 paper](https://arxiv.org/abs/2103.12591) for details, which builds on [BoXHED1.0 paper](http://proceedings.mlr.press/v119/wang20o/wang20o.pdf) (ICML 2020). The theoretical underpinnings for BoXHED is provided [here](https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-4/Boosted-nonparametric-hazards-with-time-dependent-covariates/10.1214/20-AOS2028.full) (Annals of Statistics 2021).

## Prerequisites
The software was developed and tested in Linux and Mac OS environments. The requirements are the following:
- cmake  (>=3.18.2) (Windows users: CMake is part of the Visual Studio installation)
- Python (=3.8)
- [CUDA](https://developer.nvidia.com/cuda-11.1.1-download-archive)   (=11.1) (only if GPU support is needed)
- conda  (we recommend using the free [Anaconda distribution](https://docs.anaconda.com/anaconda/install/))

Windows users need to have the Visual Studio 17 2022 toolset installed. During installation, under the "Workloads" tab select "Desktop Development with C++" in the "Desktop and Mobile" section. Make the following selections in the menu that shows up on the right:
![sc__](https://user-images.githubusercontent.com/34462617/201495851-c7d02796-31e0-4181-9eba-78065d2a5f59.png)

## [Installing BoXHED2.0 with multicore CPU support](README_CPU.md)

## [Installing BoXHED2.0 with GPU support](README_GPU.md)


We recommend setting up a dedicated virtual environment for BoXHED2.0 (instructions below). This ensures that BoXHED2.0 will not interfere with XGBoost (the library we have borrowed from extensively) when installed. This implementation uses python 3.8.

In this example we use [Anaconda Prompt](https://docs.anaconda.com/anaconda/install/) to open a terminal. First, create a virtual environment called BoXHED2.0:
```
conda create -n boxhed2.0 python=3.8
```

then activate it
```
conda activate boxhed2.0
```

Clone this repository, or manually download the files here and extract them to a directory called BoXHED2.0. Then go to the directory:
```
cd BoXHED2.0
```

Install BoXHED2.0 from pip:
```
pip install boxhed
```

Now install Jupyter Notebook:
```
pip install jupyter
```

Run *BoXHED2_tutorial.ipynb* for a demonstration of how to fit a BoXHED hazard estimator.
```
jupyter notebook BoXHED2_tutorial.ipynb
``` 

## Installing BoXHED2.0 with multicore CPU + GPU support

To add GPU support, you must first install [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive).

For Windows users, using [Anaconda Prompt](https://docs.anaconda.com/anaconda/install/) would be preferred because it eases Conda usage. For unifying the approach, we refer to the command line interface as 'terminal' but it could be Anaconda Prompt for Windows users.

We recommend setting up a dedicated virtual environment for BoXHED2.0. This ensures that BoXHED2.0 will not interfere with XGBoost (the library we have borrowed from extensively) when installed. This implementation uses python 3.8.

Open a terminal and create a virtual environment called BoXHED2.0:
```
conda create -n boxhed2.0 python=3.8
```

then activate it:
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

BoXHED2.0 needs to be built from source. First, install numpy, pandas, scikit-learn, pytz, py3nvml, matplotlib, and CudaToolkit by:
```
source conda_install_packages.sh
```

BoXHED2.0 can then be installed by running:
```
./setup.sh
```

The installer logs its activity in the terminal. If the installation is unsuccessful, check *setup_log.txt* created within the current directory for a description of the problem(s).  

In our experience, one of the most prevalent problems is that CMake cannot find the CUDA compiler, or needs some arguments set depending on your machine. If this is the case, please see the instructions at line 65 of *setup.sh* to properly set the missing arguments.

Run *BoXHED2_tutorial.ipynb* for a demonstration of how to fit a BoXHED hazard estimator.
```
jupyter notebook BoXHED2_tutorial.ipynb
``` 
