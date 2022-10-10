import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin 
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from collections.abc import Iterable
from boxhed_prep.boxhed_prep import preprocessor

#TODO: maybe this can change once I figure out the BoXHED/XGB distinction while installing

import boxhed_kernel as xgb


from boxhed_kernel import plot_tree
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


class boxhed(BaseEstimator, RegressorMixin):#ClassifierMixin, 

    def __init__(self, max_depth=1, n_estimators=100, eta=0.1, gpu_id = -1, nthread = 1):
        self.max_depth     = max_depth
        self.n_estimators  = n_estimators
        self.eta           = eta
        self.gpu_id        = gpu_id
        self.nthread       = nthread


    def _X_y_to_dmat(self, X, y=None, w=None):
        if not hasattr(self, 'X_colnames'):
            self.X_colnames = None #model probably created for CV, no need for data name matching
        dmat = xgb.DMatrix(pd.DataFrame(X, columns=self.X_colnames))

        if (y is not None):
            dmat.set_float_info('label',  y)
            dmat.set_float_info('weight', w)
    
        return dmat
        
    def preprocess(self, data, is_cat=[], num_quantiles=20, weighted=False, nthread=-1):
        self.prep = preprocessor()
        IDs, X, w, delta =  self.prep.preprocess(
            data             = data, 
            is_cat           = is_cat,
            num_quantiles    = num_quantiles, 
            weighted         = weighted, 
            nthread          = nthread)

        self.X_colnames = X.columns.values.tolist()
        self.X_colnames = [item if item!='t_start' else 'time' for item in self.X_colnames]

        return IDs, X, w, delta

    def fit (self, X, y, w=None):

        #TODO: could I do the type checking better?
        check_array(y, ensure_2d = False)
        #TODO: make sure prep exists
        # or: if does not exist, create it now and train on preprocessed

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
        return self

        
    def plot_tree(self, nom_trees):
                        
        def print_tree(i):
            print("printing tree:", i+1)
            plot_tree(self.boxhed_, num_trees = i)
            fig = plt.gcf()
            fig.set_size_inches(30, 20)
            fig.savefig("tree"+"_"+str(i+1)+'.jpg')

        for th_id in range(min(nom_trees, self.n_estimators)):
            print_tree(th_id)


    def predict(self, X, ntree_limit = 0):
        check_is_fitted(self)
        '''
        self.prep.shift_left(X)
        '''
        try:
            X = self.prep.shift_left(X)
        except:
            pass
        X = check_array(X, force_all_finite='allow-nan')

        return self.boxhed_.predict(self._X_y_to_dmat(X), ntree_limit = ntree_limit)

    def get_survival(self, X, t, ntree_limit = 0): #TODO no ind_exp
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
        preds                          = self.boxhed_.predict(cte_hazard_epoch, ntree_limit = ntree_limit)
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
