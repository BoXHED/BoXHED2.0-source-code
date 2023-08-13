import matplotlib
import matplotlib.pyplot as pl
import numpy as np
matplotlib.use('Agg')
import boxhed_shap # pylint: disable=wrong-import-position


def test_random_decision():
    """ Make sure the decision plot does not crash on random data.
    """
    np.random.seed(0)
    boxhed_shap.decision_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False)
    pl.close()

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#
# Visual tests
#
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

# import lightgbm as lgb
# import xgboost as xgb
# import matplotlib.pyplot as pl
# import numpy as np
# from scipy.special import expit
# from sklearn.model_selection import train_test_split
#
# import boxhed_shap
#
# random_state = 7
#
# X, y = boxhed_shap.datasets.adult()
# X_display, y_display = boxhed_shap.datasets.adult(display=True)
#
# # create a train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
# d_train = lgb.Dataset(X_train, label=y_train)
# d_test = lgb.Dataset(X_test, label=y_test)
#
# params = {
#     "max_bin": 512,
#     "learning_rate": 0.05,
#     "boosting_type": "gbdt",
#     "objective": "binary",
#     "metric": "binary_logloss",
#     "num_leaves": 10,
#     "verbose": -1,
#     "min_data": 100,
#     "boost_from_average": True,
#     "random_state": random_state
# }
#
# model = lgb.train(params, d_train, 1000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=False)
#
# explainer = boxhed_shap.TreeExplainer(model)
# base_value = explainer.expected_value
# select = range(20)
# features = X_test.iloc[select]
# y_label = y_test[select]
# shap_values = explainer.shap_values(features)[1]
# boxhed_shap_interaction_values = explainer.boxhed_shap_interaction_values(features)
# features_display = X_display.loc[features.index]
#
# args1 = dict(base_value=base_value, shap_values=shap_values)
# args2 = args1.copy()
# args2["shap_values"] = boxhed_shap_interaction_values
#
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# # Basic plots with default (importance) sort and generated labels.
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#
# boxhed_shap.decision_plot(**args1)
# boxhed_shap.decision_plot(**args2)
#
# boxhed_shap.decision_plot(highlight=[0, 9], **args1)
#
# boxhed_shap.decision_plot(features=features_display, **args1)
# boxhed_shap.decision_plot(features=features_display, **args2)
#
# # Plot a single observation without features
# boxhed_shap.decision_plot(base_value, shap_values[0, :])
# boxhed_shap.decision_plot(base_value, shap_values[[0], :])
#
# # Now, with a Pandas Series (and also test auto feature value positioning)
# boxhed_shap.decision_plot(base_value, shap_values[0, :], features=features_display.iloc[0, :])
# s = shap_values[0, :].copy()
# s[-1] = -35; s[-2] = 15
# boxhed_shap.decision_plot(base_value, s, features=features_display.iloc[0, :], feature_order='None')
# s[-1] = 40; s[-2] = -20
# boxhed_shap.decision_plot(base_value, s, features=features_display.iloc[0, :], feature_order='None')
# boxhed_shap.decision_plot(base_value, shap_values[4, :], features=features_display.iloc[4, :], feature_order='hclust')
# boxhed_shap.decision_plot(base_value, shap_values[7, :], features=features_display.iloc[7, :], feature_order='hclust')
# boxhed_shap.decision_plot(base_value, boxhed_shap_interaction_values[[0], :], features=features_display.iloc[0, :])
# # Now with a single observation using a matrix and a Pandas Dataframe
# boxhed_shap.decision_plot(base_value, shap_values[[0], :], features=features_display.iloc[[0], :])
# boxhed_shap.decision_plot(base_value, boxhed_shap_interaction_values[[0], :], features=features_display.iloc[[0], :])
# # Now with feature names in the features argument.
# names = features_display.columns.to_list()
# boxhed_shap.decision_plot(base_value, shap_values[[0], :], features=names)
# boxhed_shap.decision_plot(base_value, boxhed_shap_interaction_values[[0], :], features=names)
# # Now with feature names in the features argument as numpy.
# boxhed_shap.decision_plot(base_value, shap_values[[0], :], features=np.array(names))
# boxhed_shap.decision_plot(base_value, boxhed_shap_interaction_values[[0], :], features=np.array(names))
#
# names = features_display.columns.to_list()
# args1["feature_names"] = names
# args2["feature_names"] = names
#
# # Plot font changes sizes depending on whether an interaction feature is printed.
# boxhed_shap.decision_plot(feature_display_range=slice(None, -11, -1), **args2)
# boxhed_shap.decision_plot(feature_display_range=slice(None, -9, -1), **args2)
#
# # Highlighting by index
# highlight = [1, 9]
# boxhed_shap.decision_plot(highlight=highlight, **args1)
#
# # Highlighting by boolean array
# predictions = base_value + shap_values.sum(1)
# highlight = np.abs(predictions) > 9
# boxhed_shap.decision_plot(highlight=highlight, **args1)
# highlight = y_label != (expit(predictions) > 0.5)
# boxhed_shap.decision_plot(highlight=highlight, **args1)
#
# # Highlighting by slice
# boxhed_shap.decision_plot(highlight=slice(0, 10), **args1)
#
# # Logit link
# boxhed_shap.decision_plot(link="logit", **args1)
# boxhed_shap.decision_plot(link="logit", **args2)
#
# # Color scheme
# boxhed_shap.decision_plot(plot_color="coolwarm", **args1)
#
# # Axis color
# boxhed_shap.decision_plot(axis_color="#FF0000", **args1)
#
# # Y feature demarcation color
# boxhed_shap.decision_plot(y_demarc_color="#FF0000", **args1)
#
# # Alpha value
# boxhed_shap.decision_plot(alpha=0.2, **args1)
#
# # Disable color bar
# boxhed_shap.decision_plot(color_bar=False, **args1)
# boxhed_shap.decision_plot(color_bar=False, feature_display_range=slice(-20, None, 1), **args1)
#
# # Disable autosize
# boxhed_shap.decision_plot(auto_size_plot=False, **args1)
# boxhed_shap.decision_plot(auto_size_plot=False, **args2)
#
# # Enable title
# boxhed_shap.decision_plot(title="This doesn't look good", **args1)
#
# # Disable show
# boxhed_shap.decision_plot(show=False, **args1)
# pl.show()
#
# # Flip y-axis
# boxhed_shap.decision_plot(feature_display_range=slice(-20, None, 1), **args1)
# boxhed_shap.decision_plot(feature_display_range=slice(-20, None, 1), **args2)
# boxhed_shap.decision_plot(**args2) # to compare with previous plot
#
# # Use return_objects
# r = boxhed_shap.decision_plot(return_objects=True, **args1)
# idx = 8
# boxhed_shap.decision_plot(base_value, shap_values[idx], features=features_display.iloc[idx],
#                    feature_order=r.feature_idx, xlim=r.xlim)
#
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# # New base value
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#
# p = model.predict(features, raw_score=True)
#
# # boxhed_shap values w/base value zero
# new_base_value = 0
# r = boxhed_shap.decision_plot(base_value, shap_values, features, new_base_value=new_base_value, return_objects=True)
# a = r.shap_values.sum(axis=1) + new_base_value
# assert np.all(a.round(5) == p[select].round(5))
# assert r.base_value == new_base_value
#
# # boxhed_shap values w/base value non-zero
# new_base_value = 2.3
# r = boxhed_shap.decision_plot(base_value, shap_values, features, new_base_value=new_base_value, return_objects=True)
# a = r.shap_values.sum(axis=1) + new_base_value
# assert np.all(a.round(5) == p[select].round(5))
# assert r.base_value == new_base_value
#
# # boxhed_shap interaction values w/base value zero
# new_base_value = 0
# r = boxhed_shap.decision_plot(base_value, boxhed_shap_interaction_values, features, new_base_value=new_base_value,
#                        return_objects=True, feature_display_range=slice(None, None, -1))
# a = r.shap_values.sum(axis=1) + new_base_value
# assert np.all(a.round(5) == p[select].round(5))
# assert r.base_value == new_base_value
#
# # boxhed_shap interaction values w/base value non-zero
# new_base_value = -2.1
# r = boxhed_shap.decision_plot(base_value, boxhed_shap_interaction_values, features, new_base_value=new_base_value,
#                        return_objects=True, feature_display_range=slice(None, None, -1))
# a = r.shap_values.sum(axis=1) + new_base_value
# assert np.all(a.round(5) == p[select].round(5))
# assert r.base_value == new_base_value
#
# # boxhed_shap interaction values w/base value non-zero and logit link
# new_base_value = -2.1
# r = boxhed_shap.decision_plot(base_value, boxhed_shap_interaction_values, features, new_base_value=new_base_value,
#                        return_objects=True, feature_display_range=slice(None, None, -1), link='logit')
# a = r.shap_values.sum(axis=1) + new_base_value
# assert np.all(a.round(5) == p[select].round(5))
# assert r.base_value == new_base_value
#
#
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# # No sorting
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#
# boxhed_shap.decision_plot(feature_order="none", **args1)
# boxhed_shap.decision_plot(feature_order="none", feature_display_range=slice(-20, None, 1), **args1)
# boxhed_shap.decision_plot(feature_order="none", **args2)
# boxhed_shap.decision_plot(feature_order="none", feature_display_range=slice(-20, None, 1), **args2)
# boxhed_shap.decision_plot(feature_order=None, **args1)
# boxhed_shap.decision_plot(feature_order=None, **args2)
#
#
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# # Hierarchical cluster sorting
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#
# boxhed_shap.decision_plot(feature_order="hclust", **args1)
# boxhed_shap.decision_plot(feature_order="hclust", feature_display_range=slice(-20, None, 1), **args1)
# boxhed_shap.decision_plot(feature_order="hclust", **args2)
# boxhed_shap.decision_plot(feature_order="hclust", feature_display_range=slice(-20, None, 1), **args2)
#
#
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# # Feature display range
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#
# boxhed_shap.decision_plot(**args1)
# r = boxhed_shap.decision_plot(feature_display_range=range(0, 20), return_objects=True, **args1)
# boxhed_shap.decision_plot(feature_display_range=range(0, 1), xlim=r.xlim, **args1)
# boxhed_shap.decision_plot(feature_display_range=range(0, 2), xlim=r.xlim, **args1)
# boxhed_shap.decision_plot(feature_display_range=range(1, 2), xlim=r.xlim, **args1)
# boxhed_shap.decision_plot(feature_display_range=range(10, 12), xlim=r.xlim, **args1)
# boxhed_shap.decision_plot(feature_display_range=range(11, 9, -1), xlim=r.xlim, **args1)
# boxhed_shap.decision_plot(feature_display_range=range(11, 12), xlim=r.xlim, **args1)
# boxhed_shap.decision_plot(feature_display_range=range(11, 10, -1), xlim=r.xlim, **args1)
# boxhed_shap.decision_plot(feature_display_range=range(11, 0, -1), xlim=r.xlim, **args1)
#
# boxhed_shap.decision_plot(feature_display_range=slice(1), xlim=r.xlim, **args1)
# boxhed_shap.decision_plot(feature_display_range=slice(-12, -13, -1), xlim=r.xlim, **args1)
# boxhed_shap.decision_plot(feature_display_range=slice(0, 2), xlim=r.xlim, **args1)
# boxhed_shap.decision_plot(feature_display_range=slice(-11, -13, -1), xlim=r.xlim, **args1)
#
# boxhed_shap.decision_plot(feature_order='hclust', feature_display_range=slice(None, -21, -1), xlim=r.xlim, **args2)
# boxhed_shap.decision_plot(feature_order='hclust', feature_display_range=slice(20, None, -1), xlim=r.xlim, **args2)
#
# # decision_plot transforms negative values in a range so they are interpreted correctly in a slice.
# boxhed_shap.decision_plot(feature_order='hclust', feature_display_range=range(11, -1, -1), xlim=r.xlim, **args2)
# boxhed_shap.decision_plot(feature_order='hclust', feature_display_range=range(-100, 12, 1), xlim=r.xlim, **args2)
#
#
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# # Multioutput
# # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#
# X, y = boxhed_shap.datasets.iris()
# model = xgb.XGBClassifier()
# model.fit(X, y)
# explainer = boxhed_shap.TreeExplainer(model)
# sh = explainer.shap_values(X)
# ev = explainer.expected_value
# p = model.predict(X, output_margin=True)
# for i in [0, 75, 149]:
#     labels = [f'Class {j + 1} ({p[i, j]:0.2f})' for j in range(3)]
#     boxhed_shap.multioutput_decision_plot(ev, sh, i, features=X, highlight=np.argmax(p[i]), legend_labels=labels)
#
# # boxhed_shap values w/mean of expected values
# r1 = boxhed_shap.multioutput_decision_plot(ev, sh, i, features=X, highlight=np.argmax(p[i]), legend_labels=labels,
#                                return_objects=True)
# a = r1.shap_values.sum(axis=1) + np.array(ev).mean()
# print(a)
# print(p[i])
# assert np.all(a.round(5) == p[i].round(5))
# assert r1.base_value == np.array(ev).mean()
#
# # boxhed_shap values w/base value zero
# new_base_value = 0
# r1 = boxhed_shap.multioutput_decision_plot(ev, sh, i, features=X, highlight=np.argmax(p[i]), legend_labels=labels,
#                                new_base_value=new_base_value, return_objects=True)
# a = r1.shap_values.sum(axis=1) + new_base_value
# print(a)
# print(p[i])
# assert np.all(a.round(5) == p[i].round(5))
# assert r1.base_value == new_base_value
#
# # boxhed_shap interaction values w/mean of expected values
# shi = explainer.boxhed_shap_interaction_values(X)
# r1 = boxhed_shap.multioutput_decision_plot(ev, shi, i, features=X, highlight=np.argmax(p[i]), legend_labels=labels,
#                                return_objects=True)
# a = r1.shap_values.sum(axis=1) + np.array(ev).mean()
# print(a)
# print(p[i])
# assert np.all(a.round(5) == p[i].round(5))
# assert r1.base_value == np.array(ev).mean()
#
# # boxhed_shap interaction values w/base value zero
# new_base_value = 0
# r1 = boxhed_shap.multioutput_decision_plot(ev, shi, i, features=X, highlight=np.argmax(p[i]), legend_labels=labels,
#                                new_base_value=new_base_value, return_objects=True)
# a = r1.shap_values.sum(axis=1) + new_base_value
# print(a)
# print(p[i])
# assert np.all(a.round(5) == p[i].round(5))
# assert r1.base_value == new_base_value
#
# # boxhed_shap interaction values w/base value 7.5
# new_base_value = 7.5
# r1 = boxhed_shap.multioutput_decision_plot(ev, shi, i, features=X, highlight=np.argmax(p[i]), legend_labels=labels,
#                                new_base_value=new_base_value, return_objects=True)
# a = r1.shap_values.sum(axis=1) + new_base_value
# print(a)
# print(p[i])
# assert np.all(a.round(5) == p[i].round(5))
# assert r1.base_value == new_base_value
#
# # boxhed_shap interaction values w/base value 7.5 and logit link
# new_base_value = 1
# r1 = boxhed_shap.multioutput_decision_plot(ev, shi, i, features=X, highlight=np.argmax(p[i]),
#                                new_base_value=new_base_value, return_objects=True, link='logit')
# a = r1.shap_values.sum(axis=1) + new_base_value
# print(a)
# print(p[i])
# assert np.all(a.round(5) == p[i].round(5))
# assert r1.base_value == new_base_value
#
# # make sure correct feature is selected and plotted.
# idx = 1
# print(X.iloc[[idx]])
# boxhed_shap.multioutput_decision_plot([ev[0]], [sh[0]], idx, features=X, legend_labels=labels)
# boxhed_shap.multioutput_decision_plot([ev[0]], [sh[0][[idx]]], 0, features=X.iloc[idx], legend_labels=labels)
# boxhed_shap.multioutput_decision_plot([ev[0]], [sh[0]], idx, features=X.to_numpy(), legend_labels=labels)
#
