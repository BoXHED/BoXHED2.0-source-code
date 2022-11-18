## Installing BoXHED2.0 with multicore CPU support

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