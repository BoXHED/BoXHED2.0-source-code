import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin 
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from collections.abc import Iterable
from boxhed_prep.boxhed_prep import preprocessor
from . import utils

import boxhed_kernel as xgb

from boxhed_kernel import plot_tree
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


class boxhed(BaseEstimator, RegressorMixin):
    """ 
    BoXHED is a nonparametric tree-boosted hazard estimator that is fully non-parametric.
    It allows for survival settings far more general than right-censoring, including recurring events.
    For more information, see http://github.com/BoXHED/BoXHED2.0. 
    """

    def __init__(self, max_depth=1, n_estimators=100, eta=0.1, gpu_id = -1, nthread = 1):
        """BoXHED instance initializer. 

        Parameters
        ----------
        max_depth : int, optional
            The maximum depth of each tree. A tree of depth k has 2^k leaf nodes, by default 1
        n_estimators : int, optional
            Number of trees in the boosted ensemble, by default 100
        eta : float, optional
            Stepsize shrinkage, usually held fixed at a small number, by default 0.1
        gpu_id : int, optional
            GPU ID to use. Set gpu_id = -1 to use CPUs., by default -1
        nthread : int, optional
            If training with CPUs, this is the number of threads to use. Default is -1 (use all available threads)., by default 1
        """
        self.max_depth     = max_depth
        self.n_estimators  = n_estimators
        self.eta           = eta
        self.gpu_id        = gpu_id
        self.nthread       = nthread


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        out = ""
        for line in [
                "a BoXHED estimator instance",
                f"    max_depth:    {self.max_depth}",
                f"    n_estimators: {self.n_estimators}",
                f"    eta:          {self.eta}"
            ]:
            out += line+"\n"
        return out


    def _X_y_to_dmat(self, X, y=None, w=None):
        if not hasattr(self, 'X_colnames'):
            self.X_colnames = None #model probably created for CV, no need for data name matching
        dmat = xgb.DMatrix(pd.DataFrame(X, columns=self.X_colnames))

        if (y is not None):
            dmat.set_float_info('label',  y)
            dmat.set_float_info('weight', w)
    
        return dmat
        
    def preprocess(self, data, is_cat=[], split_vals={}, num_quantiles=256, weighted=False, nthread=-1):
        """
        BoXHED2.0 applies a preprocessing trick to the training data to speed up training. 
        THE DATA ONLY NEEDS TO BE PREPROCESSED ONCE PER TRAINING SET. BoXHED2.0 does not use the original training data, 
        just the outputs from the boxhed.preprocess() function.

        Parameters
        ----------
        data : pandas.DataFrame
            The data on the event histories of study subjects. The data needs to have several columns:
                * ID: subject ID
                * t_start: the start time of an epoch for the subject
                * t_end: the end time of the epoch
                * X_i: value of the i-th covariate between tstart and tend
                * delta: event label, which is 1 if an event occurred at t_end; 0 otherwise
            An illustrative example:
                +----+---------+--------+--------+-----+--------+-------+
                | ID | t_start |  t_end |    X_0 | ... |   X_10 | delta |
                +====+=========+========+========+=====+========+=======+
                |  1 |  0.0100 | 0.0747 | 0.2655 |     | 0.2059 |     1 |
                +----+---------+--------+--------+-----+--------+-------+
                |  1 |  0.0747 | 0.1072 | 0.7829 |     | 0.4380 |     0 |
                +----+---------+--------+--------+-----+--------+-------+
                |  1 |  0.1072 | 0.1526 | 0.7570 |     | 0.7789 |     1 |
                +----+---------+--------+--------+-----+--------+-------+
                |  2 |  0.2066 | 0.2105 | 0.9618 |     | 0.0859 |     1 |
                +----+---------+--------+--------+-----+--------+-------+
                |  2 |  0.2345 | 0.2716 | 0.3586 |     | 0.0242 |     0 |
                +----+---------+--------+--------+-----+--------+-------+
        is_cat : list, optional
            A list of the column indexes that contain categorical data. The categorical data must be one-hot encoded. For example, is_cat=[4,5,6] if a categorical variable with 3 factors is transformed into binary-valued columns 4,5,6., by default []
        split_vals : dict, optional
            To specify custom candidate split points for time and/or a subset of non-categorical covariates, use a dictionary to specify the values to split on (details below). Candidate split points for time/non-categorical covariates not specified in split_vals will be chosen in accordance to the num_quantiles option above., by default {}
        num_quantiles : int, optional
            the number of candidate split points to try for time and for each non-categorical covariate., by default 256
        weighted : bool, optional
            if set to True, the locations of the candidate split points will be based on weighted quantiles, by default False
        nthread : int, optional
            number of CPU threads to use for preprocessing the data., by default -1

        Returns
        -------
        dict()
            A dictionary containing the preprocessed data.
        """
        self.prep = preprocessor()
        X_post               = self.prep.preprocess(
            data             = data, 
            is_cat           = is_cat,
            split_vals       = split_vals,
            num_quantiles    = num_quantiles, 
            weighted         = weighted, 
            nthread          = nthread)

        self.train_data_cols = data.columns
        self.X_colnames = X_post['X'].columns.values.tolist()
        self.X_colnames = [item if item!='t_start' else 'time' for item in self.X_colnames]

        return X_post

    def fit (self, X, y, w=None):
        """Fit a BoXHED instance to the preprocessed training data.

        Parameters
        ----------
        X : numpy.array
            The preprocessed covariate matrix. An output of the preprocessor.
        y : numpy.array
            An indicator which equals one if event occurred at the end of epoch; zero otherwise. An output of the preprocessor.
        w : numpy.array
            Length of each epoch.  An output of the preprocessor.

        Returns
        -------
        BoXHED instance
            Returns a fitted BoXHED instance
        """

        check_array(y, ensure_2d = False)

        le = LabelEncoder()
        y  = le.fit_transform(y)
        X, y       = check_X_y(X, y, force_all_finite='allow-nan')

        if len(set(y)) <= 1:
            raise ValueError("Classifier can't train when only one class is present. All deltas are either 0 or 1.")
    
        if w is None:
            w = np.ones_like(y)

        f0_   = np.log(np.sum(y)/np.sum(w))
        dmat_ = self._X_y_to_dmat(X, y, w)

        if self.gpu_id>=0:
            self.objective_   = 'survival:boxhed_gpu'
            self.tree_method_ = 'gpu_hist'
        else:
            self.objective_   = 'survival:boxhed'
            self.tree_method_ = 'hist'

        self.params_         = {'objective':        self.objective_,
                                'tree_method':      self.tree_method_,
                                'booster':         'gbtree', 
                                'min_child_weight': 0,
                                'max_depth':        self.max_depth,
                                'eta':              self.eta,
                                'grow_policy':     'lossguide',

                                'base_score':       f0_,
                                'gpu_id':           self.gpu_id,
                                'nthread':          self.nthread
                                }
    
        self.boxhed_ = xgb.train( self.params_, 
                                  dmat_, 
                                  num_boost_round = self.n_estimators) 
        
        self.VarImps = self.boxhed_.get_score(importance_type='total_gain')
        self.time_splits = self._time_splits()
        return self

        
    def plot_tree(self, num_trees):
        """Save figures of the trees in a trained \softc instance as figures.

        Parameters
        ----------
        num_trees : int
            number of trees to plot and save to file. They will be saved in the same directory. The first *num_trees* trees will be plotted.
        """
                        
        def print_tree(i):
            print("printing tree:", i+1)
            plot_tree(self.boxhed_, num_trees = i)
            fig = plt.gcf()
            fig.set_size_inches(30, 20)
            fig.savefig("tree"+"_"+str(i+1)+'.jpg')

        for th_id in range(min(num_trees, self.n_estimators)):
            print_tree(th_id)


    def hazard(self, X, ntree_limit = 0):
        """Use the fitted BoXHED instance to estimate the hazard value for each row of the test data, which consists of a point (t, X_0, ...).

        Parameters
        ----------
        X : pandas.DataFrame
            The test data, unlike the training data, should not contain the following columns: ID, t_end, and delta. 
            An example of this data is depicted in the table below. The order of the columns should exactly match that of the training set dataframe, except for the mentioned columns that do not exist.
            +--------+--------+-----+--------+
            |      t |   X_0  | ... |  X_10  |
            +========+========+=====+========+
            | 0.0000 |   0.0  |     | 0.5081 |
            +--------+--------+-----+--------+
            | 0.0101 |   0.0  |     | 0.4149 |
            +--------+--------+-----+--------+
            | 0.0202 |   0.0  |     | 0.4077 |
            +--------+--------+-----+--------+
            | 0.0303 |   0.0  |     | 0.5897 |
            +--------+--------+-----+--------+
            | 0.0404 |   0.0  |     | 0.8405 |
            +--------+--------+-----+--------+
        ntree_limit : int, optional
            The number of trees used to make the estimation. If ntree_limit>0, the first ntree_limit trees are used for computing the output. 
            If ntree_limit is zero (set by default), all the trees are used., by default 0

        Returns
        -------
        numpy.array
            Estimated hazards. Each row of the input dataframe corresponds to one estimated hazard value.
        """
        check_is_fitted(self)

        if hasattr(self, 'prep'):
            X = self.prep.shift_left(X)

        X = check_array(X, force_all_finite='allow-nan')

        return self.boxhed_.predict(self._X_y_to_dmat(X), ntree_limit = ntree_limit)

    def _get_survival(self, X, t, ntree_limit = 0):
        def truncate_to_t(data, t):
            def _truncate_to_t(data_id):
                #data_id                   = data_id[data_id['t_start']<t]
                t_                        = data_id['t_start'].iloc[0]+t
                data_id                   = data_id[data_id['t_start']<t_]
                data_id['t_end']          = data_id['t_end'].clip(upper=t_)
                if len(data_id)>0:
                    data_id['t_end'].iloc[-1] = t_
                return data_id
            return data.groupby('ID').apply(_truncate_to_t).reset_index(drop=True)

        check_is_fitted(self)
        X                              = truncate_to_t(X, t)
        cte_hazard_epoch_df            = self.prep.epoch_break_cte_hazard(X)
        cte_hazard_epoch               = check_array(cte_hazard_epoch_df.drop(columns=["ID", "dt", "delta"]), 
                                            force_all_finite='allow-nan')
        cte_hazard_epoch               = self._X_y_to_dmat(cte_hazard_epoch)
        hzrds                          = self.boxhed_.hazard(cte_hazard_epoch, ntree_limit = ntree_limit, _shift_left=False)
        cte_hazard_epoch_df ['hzrds']  = hzrds
        cte_hazard_epoch_df ['surv']   = -cte_hazard_epoch_df ['dt'] * cte_hazard_epoch_df ['hzrds']
        surv_t                         = np.exp(cte_hazard_epoch_df.groupby('ID')['surv'].sum()).reset_index()
        surv_t.rename(columns={'surv':f'surv_at_t={t}'}, inplace=True)
        return surv_t.set_index('ID')
        


    def get_params(self, deep=None):
        """A BoXHED class getter function for obtaining parameters.

        Returns
        -------
        dict
            A dictionary containing class parameters.
        """
        return {"max_depth":     self.max_depth, 
                "n_estimators":  self.n_estimators,
                "eta":           self.eta, 
                "gpu_id":        self.gpu_id,
                "nthread":       self.nthread}


    def set_params(self, **params):
        """A BoXHED class setter function for setting parameters.

        Returns
        -------
        BoXHED instance
            Returns self after setting the parameters.
        """
        for param, val in params.items():
            setattr(self, param, val)
        return self


    def score(self, X, y, w=None, ntree_limit=0):
        """Returns a goodness-of-fit of BoXHED based on log-likelihood.

        Parameters
        ----------
        X : numpy.array
            The preprocessed covariate matrix. An output of the preprocessor.
        y : numpy.array
            An indicator which equals one if event occurred at the end of epoch; zero otherwise. An output of the preprocessor.
        w : numpy.array
            Length of each epoch.  An output of the preprocessor.
        ntree_limit : int, optional
            The number of trees used to make the estimation. If ntree_limit>0, the first ntree_limit trees are used for computing the output. 
            If ntree_limit is zero (set by default), all the trees are used., by default 0

        Returns
        -------
        float
            The log-likelihood of the input data.
        """
        X, y    = check_X_y(X, y, force_all_finite='allow-nan')
        if w is None:
            w = np.zeros_like(y)

        hzrds = self.hazard(X, ntree_limit = ntree_limit)
        return -(np.inner(hzrds, w)-np.inner(np.log(hzrds), y))

    def dump_model(self, fname):
        """Save the fitted BoXHED instance to disk.

        Parameters
        ----------
        fname : str
            The address to the file to be saved.
        """
        self.prep.prep_lib = None
        utils.dump_pickle(self, fname)
        self.prep.__init__()

    def load_model(self, fname):
        """Retrieve the fitted BoXHED instance from disk.

        Parameters
        ----------
        fname : str
            The address to the file to be retrieved.
        """
        boxhed_ = utils.load_pickle(fname)
        for attr in dir(boxhed_):
            if attr.startswith('__') or attr.startswith('_'):
                continue
            setattr(self, attr, getattr(boxhed_, attr))
        self.prep.__init__()

    def _time_splits(self):
        trees_df = self.boxhed_.trees_to_dataframe()
        return np.sort(np.unique(trees_df[trees_df['Feature']=='time']['Split'].values))

    def survivor(self, X, ntree_limit = 0):
        """The survivor curve is not meaningful when the covariates change over time. However, if they are static, S(t|x) can be estimated using BoXHED.

        Parameters
        ----------
        X : pandas.DataFrame
            A dataframe where only the valye 't' changes. Make sure t is monotonically increasing.
            +------+--------+-----+--------+
            |   t  |   X_0  | ... |  X_10  |
            +======+========+=====+========+
            | 0.00 | 0.2655 |     | 0.2059 |
            +------+--------+-----+--------+
            | 0.01 | 0.2655 |     | 0.2059 |
            +------+--------+-----+--------+
            | 0.02 | 0.2655 |     | 0.2059 |
            +------+--------+-----+--------+
            | 0.03 | 0.2655 |     | 0.2059 |
            +------+--------+-----+--------+
            | 0.04 | 0.2655 |     | 0.2059 |
            +------+--------+-----+--------+
        ntree_limit : int, optional
            The number of trees used to make the estimation. If ntree_limit>0, the first ntree_limit trees are used for computing the output. 
            If ntree_limit is zero (set by default), all the trees are used., by default 0

        Returns
        -------
        numpy.array
            A Numpy array of survivor values at each point in time in the input.
        """
        check_is_fitted(self)
        X                               = X.rename(columns={'t':'t_end', 'time':'t_end'})
        t_zero_idxs                     = np.where(X['t_end'].values==0)[0]
        X['t_end'].replace(0, sys.float_info.epsilon, inplace=True)
        X['t_start']                    = 0
        X['delta']                      = 0
        X['ID']                         = range(1, X.shape[0]+1)
        X                               = X[self.train_data_cols]

        self.prep.update_time_splits (self.time_splits)

        cte_hazard_epoch_df             = self.prep.epoch_break_cte_hazard(X)
        cte_hazard_epoch_df['t_start']  = cte_hazard_epoch_df['t_start'] + 0.5 * cte_hazard_epoch_df['dt']

        cte_hazard_epoch                = cte_hazard_epoch_df.drop(columns=["ID", "dt", "delta"])
        hzrds                           = self.hazard(cte_hazard_epoch, ntree_limit = ntree_limit)
        cte_hazard_epoch_df ['hzrds']   = hzrds
        cte_hazard_epoch_df ['surv']    = -cte_hazard_epoch_df ['dt'] * cte_hazard_epoch_df ['hzrds']

        survs                           = np.exp(cte_hazard_epoch_df.groupby('ID')['surv'].sum()).values
        survs[t_zero_idxs]              = 1
        return survs