## Installing BoXHED2.0 with GPU support

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
