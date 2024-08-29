# BoXHED2.0

**B**oosted e**X**act **H**azard **E**stimator with **D**ynamic covariates v2.0 (BoXHED2.0, pronounced 'box-head') is a software package designed for nonparametric estimation of hazard functions using gradient-boosted trees. BoXHED2.0 supports both time-static and time-dependent covariates. 

This document provides instructions for manually installing the [BoXHED2.0](https://github.com/BoXHED/BoXHED2.0) package. For additional information, including tutorials and related publications, please visit the package's GitHub page.

## Prerequisites
BoXHED2.0 has been developed and tested on Linux and macOS. The following are required:
- cmake  (>=3.18.2) (Note for Windows users: CMake is included in the Visual Studio installation.)
- Python (=3.8)
- [CUDA](https://developer.nvidia.com/cuda-11.1.1-download-archive)   (=11.1) (Required only if GPU support is needed.)
- conda  (we recommend using the free [Anaconda distribution](https://docs.anaconda.com/anaconda/install/))

For Windows users, the Visual Studio 17 2022 toolset must be installed. During installation, select "Desktop Development with C++" under the "Workloads" tab in the "Desktop and Mobile" section. Ensure the following options are selected:
![sc__](https://user-images.githubusercontent.com/34462617/201495851-c7d02796-31e0-4181-9eba-78065d2a5f59.png)

## [Installing BoXHED2.0 (multicore CPU support)](README_CPU.md)

## [Installing BoXHED2.0 (multicore CPU + GPU support)](README_GPU.md)

## [Installing SHAP for BoXHED](README_SHAP.md)
