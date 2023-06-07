## Installing SHAP for BoXHED

`BoXHED2.0` can harness the power of Shapley values in explaining the effect of time and covariates on the predicted log-hazard. To attain Shapley values, `BoXHED2.0` relies heavily on the Python package `shap` version 0.41.0. For more information, refer to their [documentation page](https://shap.readthedocs.io/en/latest/).


We recommend using the already dedicated virtual environment, as instructed previously, for installing the `shap` package.

First we need to activate it:
```
conda activate boxhed2.0
```

Assuming the current directory is that of the cloned repository, install the `shap` package using `pip`:
```
pip install ./packages/boxhed_shap
```

Instructions on using `shap` can be found in *BoXHED2_tutorial.ipynb*.
```
jupyter notebook BoXHED2_tutorial.ipynb
``` 
