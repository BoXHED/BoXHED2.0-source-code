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


class boxhed(BaseEstimator, RegressorMixin):#ClassifierMixin, 

    def __init__(self, max_depth=1, n_estimators=100, eta=0.1, gpu_id = -1, nthread = 1):
        """initializer of the BoXHED class.

        :param max_depth: maximum depth of each tree. Depth $l$ hosts $2^l$ terminal nodes., defaults to 1
        :type max_depth: int, optional
        :param n_estimators: number of trees., defaults to 100
        :type n_estimators: int, optional
        :param eta: the learning rate., defaults to 0.1
        :type eta: float, optional
        :param gpu_id: the ID of the GPU to be used. Set to -1 (default value) to use CPU., defaults to -1
        :type gpu_id: int, optional
        :param nthread: the number of CPU threads to be used., defaults to 1
        :type nthread: int, optional
        """
        self.max_depth     = max_depth
        self.n_estimators  = n_estimators
        self.eta           = eta
        self.gpu_id        = gpu_id
        self.nthread       = nthread


    def X_y_to_dmat(self, X, y=None, w=None):
        if not hasattr(self, 'X_colnames'):
            self.X_colnames = None #model probably created for CV, no need for data name matching
        dmat = xgb.DMatrix(pd.DataFrame(X, columns=self.X_colnames))

        if (y is not None):
            dmat.set_float_info('label',  y)
            dmat.set_float_info('weight', w)
    
        return dmat
        
    def preprocess(self, data, is_cat=[], split_vals={}, num_quantiles=256, weighted=False, nthread=-1):
        """preprocess the training data before fitting a BoXHED instance.

        :param data: input training data. 
        ID	t_start     t_end       X_0         delta
        1   0.010000    0.064333    0.152407	0.0
        1   0.064333    0.135136    0.308475	0.0
        1   0.194810    0.223106    0.614977	1.0
        1   0.223106    0.248753    0.614977	0.0
        2   0.795027    0.841729    0.196407	1.0
        2   0.841729    0.886587    0.196407	0.0
        2   0.886587    0.949803    0.671227	0.0
        Each row corresponds to an epoch in patient history. Column \textit{ID} denotes the patient identifier. The start and end times of the epoch are stored in \textit{t\_start} and \textit{t\_end} respectively. Each change in their status marks a new epoch. For obvious reasons, t\_start < t\_end for all epochs. Also, no epoch can start earlier than the end of the previous one for the same patient, i.e. ${t\_end}_i <= {t\_start}_{i+1}$. Column $X_0$ records the values of covariate $X_0$. Finally, the value in column \textit{delta} equals $1$ if the epoch ends with the event (possibly recurring) and equals $0$ otherwise. \softc expects the input Pandas dataframe to have columns with these exact names: \textit{ID}, \textit{t\_start}, \textit{t\_end}, and \textit{delta}. All other columns (if any) will be interpreted as covariates.
        :type data: pd.DataFrame
        :param is_cat: a list of the column indexes that contain categorical data. The categorical data must be one-hot encoded. For example, is\_cat = [4,5,6] if a categorical variable with 3 factors is transformed into binary-valued columns 4,5,6., defaults to []
        :type is_cat: list, optional
        :param split_vals: a dictionary to specify values to split on for any covariate or time. The key should the variable name and the value a list (or a 1d NumPy array) containing candidate split points. For specifying candidate points for time the key value should simply be 'time' and for other covariates it should exactly match the column name in the dataset. Key values 'ID', 't_start', 't_end', and 'delta' are not allowed. If the candidate split points of a covariate/time is not specified in split_vals, extracted quantiles (the default behavior) would be used. Be sure to not specify split points whose number exceeds the num_quantiles variable., defaults to {}
        :type split_vals: dict, optional
        :param num_quantiles: the number of candidate split points to try for time and for each covariate. The locations of the split points are based on the quantiles of the training data., defaults to 256
        :type num_quantiles: int, optional
        :param weighted: if set to True, the locations of the candidate split points will be based on weighted quantiles., defaults to False
        :type weighted: bool, optional
        :param nthread: number of CPU threads to use for preprocessing the data., defaults to -1
        :type nthread: int, optional
        :return: 
                \textbf{ID}: subject ID for each row in the processed data frames X, w, and delta.
                
                \textbf{X}: each row represents an epoch of the transformed data, and contains the values of the covariates as well as its start time.
                
                \textbf{w}: length of each epoch.
                
                \textbf{delta}: equals one if an event occurred at the end of the epoch; zero otherwise.
        :rtype: pd.Series, pd.DataFrame, pd.Series, pd.Series
        """
        self.prep = preprocessor()
        X_post               = self.prep.preprocess(
            data             = data, 
            is_cat           = is_cat,
            split_vals       = split_vals,
            num_quantiles    = num_quantiles, 
            weighted         = weighted, 
            nthread          = nthread)

        self.X_colnames = X_post['X'].columns.values.tolist()
        self.X_colnames = [item if item!='t_start' else 'time' for item in self.X_colnames]

        return X_post

    def fit (self, X, y, w=None):
        """_summary_

        :param X: the preprocessed covariate matrix. An output of the preprocessor.
        :type X: pd.DataFrame
        :param y: length of each epoch.  An output of the preprocessor.
        :type y: pd.Series
        :param w: length of each epoch.  An output of the preprocessor.
        :type w: pd.Series
        :return: trained BoXHED instance
        :rtype: BoXHED estimator
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
        dmat_ = self.X_y_to_dmat(X, y, w)

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
        return self

        
    def plot_tree(self, num_trees):
        """saving figures of the trees in a trained \softc instance as figures.

        :param num_trees: number of trees to plot and save to file. They will be saved in the same directory. The first \textit{num_trees} trees will be plotted.
        :type num_trees: int
        """
                        
        def print_tree(i):
            print("printing tree:", i+1)
            plot_tree(self.boxhed_, num_trees = i)
            fig = plt.gcf()
            fig.set_size_inches(30, 20)
            fig.savefig("tree"+"_"+str(i+1)+'.jpg')

        for th_id in range(min(num_trees, self.n_estimators)):
            print_tree(th_id)


    def predict(self, X, ntree_limit = 0, _shift_left=True):
        """_summary_

        :param X: a Pandas dataframe. The test data, unlike the training data, should not contain the following columns: \textit{ID}, \textit{t\_end}, and \textit{delta}. An example of this data is depicted in a table below. The order of the columns should exactly match that of the training set dataframe, except for the mentioned columns that do not exist.
        t       X0
        0.01    0.15
        0.06    0.30
        0.89    0.67
        0.19    0.61
        0.22    0.61
        0.80    0.19
        :type X: pd.DataFrame
        :param ntree_limit: _description_, defaults to 0
        :type ntree_limit: int, optional
        :return: a NumPy array which is a column vector. The $i$'th element corresponds to the $i$'th row in \textbf{X}.
        :rtype: np.array
        """
        check_is_fitted(self)
        if _shift_left:
            if hasattr(self, 'prep'):
                X = self.prep.shift_left(X)

        X = check_array(X, force_all_finite='allow-nan')

        return self.boxhed_.predict(self.X_y_to_dmat(X), ntree_limit = ntree_limit)

    def get_survival(self, X, t, ntree_limit = 0):
        """estimating survival probability at a specific time using a trained BoXHED instance.

        :param X: a Pandas dataframe. The structure exactly follows the one boxhed.fit() uses.
        :type X: pd.DataFrame
        :param t: the time at which survival needs to be calculated. The survival probability of all subjects is calculated for the same time \textit{t}.
        :type t: float
        :param ntree_limit: the number of trees used to make the estimation. If \textit{ntree\_limit}>0, the first \textit{ntree\_limit} trees are used for computing the output. If \textit{ntree\_limit} is zero (set by default), all of the trees are used., defaults to 0
        :type ntree_limit: int, optional
        """
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
        cte_hazard_epoch               = self.X_y_to_dmat(cte_hazard_epoch)
        preds                          = self.boxhed_.predict(cte_hazard_epoch, ntree_limit = ntree_limit, _shift_left=False)
        cte_hazard_epoch_df ['preds']  = preds
        cte_hazard_epoch_df ['surv']   = -cte_hazard_epoch_df ['dt'] * cte_hazard_epoch_df ['preds']
        surv_t                         = np.exp(cte_hazard_epoch_df.groupby('ID')['surv'].sum()).reset_index()
        surv_t.rename(columns={'surv':f'surv_at_t={t}'}, inplace=True)
        return surv_t.set_index('ID')
        


    def get_params(self, deep=True):
        return {"max_depth":     self.max_depth, 
                "n_estimators":  self.n_estimators,
                "eta":           self.eta, 
                "gpu_id":        self.gpu_id,
                "nthread":       self.nthread}


    def set_params(self, **params):
        for param, val in params.items():
            setattr(self, param, val)
        return self


    def score(self, X, y, w=None, ntree_limit=0):
        X, y    = check_X_y(X, y, force_all_finite='allow-nan')
        if w is None:
            w = np.zeros_like(y)

        preds = self.predict(X, ntree_limit = ntree_limit)
        return -(np.inner(preds, w)-np.inner(np.log(preds), y))

    def dump_model(self, fname):
        self.prep.prep_lib = None
        utils.dump_pickle(self, fname)
        self.prep.__init__()

    def load_model(self, fname):
        boxhed_ = utils.load_pickle(fname)
        for attr in dir(boxhed_):
            if attr.startswith('__') or attr.startswith('_'):
                continue
            setattr(self, attr, getattr(boxhed_, attr))
        self.prep.__init__()


    def time_splits(self):
        check_is_fitted(self)
        trees_df = self.boxhed_.trees_to_dataframe()
        return np.sort(trees_df[trees_df['Feature']=='time']['Split'].values)
