## Installing SHAP for BoXHED

Another variable importance measure called SHAP values (SHapley Additive exPlanations) can be calculated for the `BoXHED2.0` log-hazard function. For this we rely heavily on the Python package `shap` version 0.41.0. For more information, refer to their [documentation page](https://shap.readthedocs.io/en/latest/).

To use SHAP with `BoXHED2.0`, first activate the dedicated virtual environment for `BoXHED2.0`, and then go to the directory containing the cloned repository: 
```
conda activate boxhed2
cd BoXHED2
```

Install the `shap` package using `pip`:
```
pip install ./packages/boxhed_shap
```
