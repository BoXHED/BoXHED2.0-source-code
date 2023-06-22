## Installing BoXHED2.0 (multicore CPU + GPU support)

To add GPU support, you must first install [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive). Note that CUDA is only available on Linux or Windows, as Macs do not use Nvidia GPUs.

For Windows users, use the [Git Bash](https://gitforwindows.org/) terminal for the installation. You will need to [set up conda in Git Bash](https://discuss.codecademy.com/t/setting-up-conda-in-git-bash/534473). Linux users should use the default terminal.

Install Visual Studio before attempting the below. See the [Prerequisites](https://github.com/BoXHED/BoXHED2.0/) section of the main page.

Next, set up a dedicated virtual environment for BoXHED2.0. This ensures that BoXHED2.0 will not interfere with any existing XGBoost packages. This implementation uses python 3.8.

Open a terminal and create a virtual environment called BoXHED2.0:
```
conda create -n boxhed2 python=3.8
```

then activate it:
```
conda activate boxhed2
```

Clone this repository (or manually download the files and extract them to a directory called BoXHED2):
```
git clone https://github.com/BoXHED/BoXHED2.0.git
```

Then, go to the directory:
```
cd BoXHED2
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

BoXHED2.0 also supports the use of SHAP values. Please see the installation instructions for the [BoXHED SHAP](https://github.com/BoXHED/BoXHED2.0/blob/master/README_SHAP.md) package.
