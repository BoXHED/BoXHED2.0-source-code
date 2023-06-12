## Installing SHAP for BoXHED

Another variable importance measure called SHAP values (SHapley Additive exPlanations) can be calculated for the `BoXHED2.0` log-hazard function. For this we rely heavily on the Python package `shap` version 0.41.0. For more information, refer to their [documentation page](https://shap.readthedocs.io/en/latest/).

To use SHAP with `BoXHED2.0`, first activate the dedicated virtual environment for `BoXHED2.0`, and then go to the directory containing the cloned repository: 
```
conda activate boxhed2.0
cd BoXHED2.0
```

Install the `shap` package using `pip`:
```
pip install ./packages/boxhed_shap
```

The tutorial *BoXHED2_tutorial.ipynb* provides example code for how to use `shap`:
```
jupyter notebook BoXHED2_tutorial.ipynb
```

To save disk space, you may delete the directory `./packages/boxhed_shap` and its contents (~400Mb) after installation. 
