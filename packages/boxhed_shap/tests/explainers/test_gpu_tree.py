# pylint: disable=missing-function-docstring
""" Test gpu accelerated tree functions.
"""
import sklearn
import pytest
import numpy as np
import boxhed_shap
from boxhed_shap.utils import assert_import

try:
    assert_import("cext_gpu")
except ImportError:
    pytestmark = pytest.mark.skip("cuda module not built")


def test_front_page_xgboost():
    xgboost = pytest.importorskip("xgboost")

    # load JS visualization code to notebook
    boxhed_shap.initjs()

    # train XGBoost model
    X, y = boxhed_shap.datasets.california(n_points=500)
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

    # explain the model's predictions using boxhed_shap values
    explainer = boxhed_shap.GPUTreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explaination
    boxhed_shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

    # visualize the training set predictions
    boxhed_shap.force_plot(explainer.expected_value, shap_values, X)

    # create a boxhed_shap dependence plot to show the effect of a single feature across the whole dataset
    boxhed_shap.dependence_plot(5, shap_values, X, show=False)
    boxhed_shap.dependence_plot("Longitude", shap_values, X, show=False)

    # summarize the effects of all the features
    boxhed_shap.summary_plot(shap_values, X, show=False)


rs = np.random.RandomState(15921)  # pylint: disable=no-member
n = 100
m = 4
datasets = {'regression': (rs.randn(n, m), rs.randn(n)),
            'binary': (rs.randn(n, m), rs.binomial(1, 0.5, n)),
            'multiclass': (rs.randn(n, m), rs.randint(0, 5, n))}


def task_xfail(func):
    def inner():
        return pytest.param(func(), marks=pytest.mark.xfail)

    return inner


def xgboost_base():
    # pylint: disable=import-outside-toplevel
    try:
        import xgboost
    except ImportError:
        return pytest.param("xgboost.XGBRegressor", marks=pytest.mark.skip)
    X, y = datasets['regression']

    model = xgboost.XGBRegressor(tree_method="hist")
    model.fit(X, y)
    return model.get_booster(), X, model.predict(X)


def xgboost_regressor():
    # pylint: disable=import-outside-toplevel
    try:
        import xgboost
    except ImportError:
        return pytest.param("xgboost.XGBRegressor", marks=pytest.mark.skip)

    X, y = datasets['regression']

    model = xgboost.XGBRegressor()
    model.fit(X, y)
    return model, X, model.predict(X)


def xgboost_binary_classifier():
    # pylint: disable=import-outside-toplevel
    try:
        import xgboost
    except ImportError:
        return pytest.param("xgboost.XGBClassifier", marks=pytest.mark.skip)

    X, y = datasets['binary']

    model = xgboost.XGBClassifier(eval_metric='error')
    model.fit(X, y)
    return model, X, model.predict(X, output_margin=True)


def xgboost_multiclass_classifier():
    # pylint: disable=import-outside-toplevel
    try:
        import xgboost
    except ImportError:
        return pytest.param("xgboost.XGBClassifier", marks=pytest.mark.skip)

    X, y = datasets['multiclass']

    model = xgboost.XGBClassifier()
    model.fit(X, y)
    return model, X, model.predict(X, output_margin=True)


def lightgbm_base():
    # pylint: disable=import-outside-toplevel
    try:
        import lightgbm
    except ImportError:
        return pytest.param("lightgbm.LGBMRegressor", marks=pytest.mark.skip)
    X, y = datasets['regression']

    model = lightgbm.LGBMRegressor()
    model.fit(X, y)
    return model.booster_, X, model.predict(X)


def lightgbm_regression():
    # pylint: disable=import-outside-toplevel
    try:
        import lightgbm
    except ImportError:
        return pytest.param("lightgbm.LGBMRegressor", marks=pytest.mark.skip)
    X, y = datasets['regression']

    model = lightgbm.LGBMRegressor()
    model.fit(X, y)
    return model, X, model.predict(X)


def lightgbm_binary_classifier():
    # pylint: disable=import-outside-toplevel
    try:
        import lightgbm
    except ImportError:
        return pytest.param("lightgbm.LGBMClassifier", marks=pytest.mark.skip)
    X, y = datasets['binary']

    model = lightgbm.LGBMClassifier()
    model.fit(X, y)
    return model, X, model.predict(X, raw_score=True)


def lightgbm_multiclass_classifier():
    # pylint: disable=import-outside-toplevel
    try:
        import lightgbm
    except ImportError:
        return pytest.param("lightgbm.LGBMClassifier", marks=pytest.mark.skip)
    X, y = datasets['multiclass']

    model = lightgbm.LGBMClassifier()
    model.fit(X, y)
    return model, X, model.predict(X, raw_score=True)


def rf_regressor():
    X, y = datasets['regression']
    model = sklearn.ensemble.RandomForestRegressor()
    model.fit(X, y)
    return model, X, model.predict(X)


def rf_binary_classifier():
    X, y = datasets['binary']
    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(X, y)
    return model, X, model.predict_proba(X)


def rf_multiclass_classifier():
    X, y = datasets['multiclass']
    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(X, y)
    return model, X, model.predict_proba(X)


tasks = [xgboost_base(), xgboost_regressor(), xgboost_binary_classifier(),
         xgboost_multiclass_classifier(), lightgbm_base(), lightgbm_regression(),
         lightgbm_binary_classifier(), lightgbm_multiclass_classifier(), rf_binary_classifier(),
         rf_regressor(), rf_multiclass_classifier()]


# pretty print tasks
def idfn(task):
    if isinstance(task, str):
        return task
    model, _, _ = task
    return type(model).__module__ + '.' + type(model).__qualname__


@pytest.mark.parametrize("task", tasks, ids=idfn)
@pytest.mark.parametrize("feature_perturbation", ["interventional", "tree_path_dependent"])
def test_gpu_tree_explainer_boxhed_shap(task, feature_perturbation):
    model, X, _ = task
    gpu_ex = boxhed_shap.GPUTreeExplainer(model, X, feature_perturbation=feature_perturbation)
    ex = boxhed_shap.TreeExplainer(model, X, feature_perturbation=feature_perturbation)
    host_boxhed_shap = ex.shap_values(X, check_additivity=True)
    gpu_boxhed_shap = gpu_ex.shap_values(X, check_additivity=True)

    # Check outputs roughly the same as CPU algorithm
    assert np.allclose(ex.expected_value, gpu_ex.expected_value, 1e-3, 1e-3)
    assert np.allclose(host_boxhed_shap, gpu_boxhed_shap, 1e-3, 1e-3)


@pytest.mark.parametrize("task", tasks, ids=idfn)
@pytest.mark.parametrize("feature_perturbation", ["tree_path_dependent"])
def test_gpu_tree_explainer_boxhed_shap_interactions(task, feature_perturbation):
    model, X, margin = task
    ex = boxhed_shap.GPUTreeExplainer(model, X, feature_perturbation=feature_perturbation)
    shap_values = np.array(ex.boxhed_shap_interaction_values(X), copy=False)

    assert np.abs(np.sum(shap_values, axis=(len(shap_values.shape) - 1, len(
        shap_values.shape) - 2)).T + ex.expected_value - margin).max() < 1e-4, \
        "boxhed_shap values don't sum to model output!"
