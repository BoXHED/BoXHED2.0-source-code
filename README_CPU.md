## Installing BoXHED2.0 (multicore CPU support)

Install Visual Studio before attempting the below. See the [Prerequisites](https://github.com/BoXHED/BoXHED2.0/) section of the main page.

Next, set up a dedicated virtual environment for BoXHED2.0. This ensures that BoXHED2.0 will not interfere with any existing XGBoost packages. This implementation uses python 3.8.

In this example we use [Anaconda Prompt](https://docs.anaconda.com/anaconda/install/) to open a terminal. First, create a virtual environment called BoXHED2.0:
```
conda create -n boxhed2 python=3.8
```

then activate it
```
conda activate boxhed2
```

Clone this repository, or manually download the files and extract them to a directory called BoXHED2.0. Then go to the directory:
```
cd BoXHED2
```

Install the dependencies by pasting the following lines in your terminal:
```
pip install matplotlib==3.7.1
pip install pillow==9.4.0
pip install numpy==1.24.3
pip install scikit-learn==1.2.2
pip install pytz==2023.3
pip install pandas==1.5.3
pip install cmake==3.26.3
pip install py3nvml==0.2.7
pip install tqdm==4.65.0
pip install threadpoolctl==3.1.0
pip install scipy==1.10.1
pip install joblib==1.2.0
pip install --force-reinstall --upgrade python-dateutil
pip install jupyter
```

Install BoXHED2.0 from pip:
```
pip install boxhed
```
