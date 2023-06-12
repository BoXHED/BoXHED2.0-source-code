## Installing BoXHED2.0 (multicore CPU support)

We recommend setting up a dedicated virtual environment for BoXHED2.0 (instructions below). This ensures that BoXHED2.0 will not interfere with XGBoost (the library we have borrowed from extensively) when installed. This implementation uses python 3.8.

In this example we use [Anaconda Prompt](https://docs.anaconda.com/anaconda/install/) to open a terminal. First, create a virtual environment called BoXHED2.0:
```
conda create -n boxhed2.0 python=3.8
```

then activate it
```
conda activate boxhed2.0
```

Clone this repository, or manually download the files and extract them to a directory called BoXHED2.0. Then go to the directory:
```
cd BoXHED2.0
```

Install the dependencies by pasting the following lines in your terminal:
```
conda install -c anaconda pandas=1.5.2 numpy=1.24.3 scikit-learn=1.2.2 pytz=2022.7 jupyter -y
conda install -c conda-forge matplotlib=3.7.1 -y
pip install pandas==1.5.2
pip install cmake==3.26.3
pip install py3nvml==0.2.7
pip install jupyter
```

Install BoXHED2.0 from pip:
```
pip install boxhed
```

Run *BoXHED2_tutorial.ipynb* for a demonstration of how to fit a BoXHED hazard estimator.
```
jupyter notebook BoXHED2_tutorial.ipynb
``` 
