""" Unit tests for the Sampling explainer.
"""

# pylint: disable=missing-function-docstring
import numpy as np
import pytest
import boxhed_shap


def test_null_model_small():

    explainer = boxhed_shap.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), nsamples=100)
    shap_values = explainer.shap_values(np.ones((1, 4)))
    assert np.sum(np.abs(shap_values)) < 1e-8


def test_null_model_small_new():

    explainer = boxhed_shap.explainers.Sampling(lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), nsamples=100)
    shap_values = explainer(np.ones((1, 4)))
    assert np.sum(np.abs(shap_values.values)) < 1e-8


def test_null_model():

    explainer = boxhed_shap.SamplingExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 10)), nsamples=100)
    shap_values = explainer.shap_values(np.ones((1, 10)))
    assert np.sum(np.abs(shap_values)) < 1e-8


def test_front_page_model_agnostic():
    sklearn = pytest.importorskip('sklearn')
    train_test_split = pytest.importorskip('sklearn.model_selection').train_test_split

    # print the JS visualization code to the notebook
    boxhed_shap.initjs()

    # train a SVM classifier
    X_train, X_test, Y_train, _ = train_test_split(*boxhed_shap.datasets.iris(), test_size=0.2, random_state=0)
    svm = sklearn.svm.SVC(kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel boxhed_shap to explain test set predictions
    explainer = boxhed_shap.SamplingExplainer(svm.predict_proba, X_train, nsamples=100)
    explainer.shap_values(X_test)
