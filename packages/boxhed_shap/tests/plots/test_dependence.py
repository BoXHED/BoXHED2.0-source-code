import matplotlib
import numpy as np
matplotlib.use('Agg')
import boxhed_shap # pylint: disable=wrong-import-position

def test_random_dependence():
    """ Make sure a dependence plot does not crash.
    """
    boxhed_shap.dependence_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False)

def test_random_dependence_no_interaction():
    """ Make sure a dependence plot does not crash when we are not showing interations.
    """
    boxhed_shap.dependence_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False, interaction_index=None)
