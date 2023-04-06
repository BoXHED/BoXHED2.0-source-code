from boxhed.boxhed import boxhed


from sklearn.utils import indexable
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
import numpy as np
import copy
from tqdm import tqdm
#'''
import os
#TODO: CAN I DO ANYTHING ABOUT OMP_NUM_THREADS
os.environ['OMP_NUM_THREADS'] = "1"
#'''
from itertools import product
from multiprocessing import Manager
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from sklearn.model_selection import GroupKFold 



def _to_shared_mem(smm, arr_np):
    shared_mem       = smm.SharedMemory(size=arr_np.nbytes)
    arr_np_shared    = np.ndarray(shape = arr_np.shape, dtype = arr_np.dtype, buffer = shared_mem.buf)
    arr_np_shared[:] = arr_np[:]
    return shared_mem


def _run_batch_process(param_dict_, rslts):

    #[X, y] = child_data_conn.recv()

    param_dicts_train = param_dict_['param_dicts_train']
    param_dicts_test  = param_dict_['param_dicts_test']
    data_idx          = param_dict_['data_idx']
    X_shared_name     = param_dict_['X_shared_name']
    X_shape           = param_dict_['X_shape']
    X_dtype           = param_dict_['X_dtype']
    w_shared_name     = param_dict_['w_shared_name']
    w_shape           = param_dict_['w_shape']
    w_dtype           = param_dict_['w_dtype']
    delta_shared_name = param_dict_['delta_shared_name']
    delta_shape       = param_dict_['delta_shape']
    delta_dtype       = param_dict_['delta_dtype']
    batch_size        = param_dict_['batch_size']
    batch_idx         = param_dict_['batch_idx']
    test_block_size   = param_dict_['test_block_size']

    smem_x     = SharedMemory(X_shared_name)
    smem_w     = SharedMemory(w_shared_name)
    smem_delta = SharedMemory(delta_shared_name)

    X     = np.ndarray(shape = X_shape, dtype = X_dtype, buffer = smem_x.buf)
    w     = np.ndarray(shape = w_shape, dtype = w_dtype, buffer = smem_w.buf)
    delta = np.ndarray(shape = delta_shape, dtype = delta_dtype, buffer = smem_delta.buf)



    def _fit_single_model(param_dict):
        estimator = boxhed()
        estimator.set_params(**param_dict)

        fold = param_dict["fold"]
        estimator.fit(X    [data_idx[fold]['train'], :], 
                      delta[data_idx[fold]['train']],
                      w    [data_idx[fold]['train']])

        return estimator


    def _get_score_batch_est_test(idx_dict):
        est_idx   = idx_dict['est_idx']
        test_idx  = idx_dict['test_idx']
        abs_idx   = batch_idx*batch_size+est_idx
        test_dict = param_dicts_test[test_block_size*abs_idx + test_idx]

        n_trees = test_dict['n_estimators']
        fold    = test_dict['fold']

        est       = trained_models[est_idx]
        
        return est.score(
            X    [data_idx[fold]['test'], :], 
            delta[data_idx[fold]['test']],
            w    [data_idx[fold]['test']],
            ntree_limit = n_trees)

    def _get_score_batch_est(idx_dicts):
        scores = [None] * len(idx_dicts)
        for i, idx_dict in enumerate(idx_dicts):
            scores[i] = _get_score_batch_est_test(idx_dict)
        return scores


    def _fill_rslt():
        batch_block_size = test_block_size*len(trained_models)
        idx_dicts = [{"est_idx": rslt_idx // test_block_size, "test_idx":rslt_idx % test_block_size} 
            for rslt_idx in range(batch_block_size)]
        idx_dicts = [idx_dicts[i : i + test_block_size] for i in range(0, len(idx_dicts), test_block_size)]
        scores = Parallel(n_jobs = batch_size, prefer = "threads")(delayed (_get_score_batch_est)(idx_dicts_) 
            for idx_dicts_ in idx_dicts)
        scores = [item for sublist in scores for item in sublist]
        rslts [batch_block_size*batch_idx:batch_block_size*(batch_idx+1)] = scores

    
    trained_models = Parallel(n_jobs=-1, prefer="threads")(delayed(_fit_single_model)(param_dict) 
            for param_dict in param_dicts_train[batch_idx*batch_size:
                (batch_idx+1)*batch_size])

    _fill_rslt()


class collapsed_gs_:

    def _remove_keys(self, dict_, keys):
        dict_ = copy.copy(dict_)
        for key in keys:
            dict_.pop(key, None)
        return dict_

    def _dict_to_tuple(self, dict_):
        list_ = []
        for key in sorted(dict_.keys()):
            list_ += [key, dict_[key]]
        return tuple(list_)


    def _fill_data_idx(self):
        for idx, (train_idx, test_idx) in enumerate(self.cv):
            self.data_idx[idx] = {"train":train_idx ,"test":test_idx}
            

    def __init__(self, param_grid, cv, GPU_LIST, batch_size):

        self.param_grid = copy.copy(param_grid)
        self.param_grid["fold"] = [x for x in range(len(cv))]

        #self.model_per_gpu = model_per_gpu
        self.batch_size = batch_size
        self.model_per_gpu = batch_size//len(GPU_LIST)
        
        self.collapsed     = 'n_estimators'
        self.not_collapsed = 'max_depth'

        self.param_grid_train = copy.copy(self.param_grid)

        self.param_grid_train[self.collapsed] = [
                max(self.param_grid_train[self.collapsed])
                ]

        self.GPU_LIST = GPU_LIST

        self.param_dicts_train = self._make_indiv_dicts(
                self.param_grid_train, train=True)       

        self.param_dicts_test = self._make_indiv_dicts(self.param_grid)

        self.param_dicts_test.sort(key=lambda dict_: 
                (dict_[self.not_collapsed], dict_['fold']))

        self.test_block_size = len(self.param_grid[self.collapsed])

        self.cv         = cv

        self.data_idx   = {}
        self._fill_data_idx()

        manager                = Manager()
        self.rslts             = manager.list(['0']*len(self.param_dicts_test))


    def _make_indiv_dicts(self, dict_, train=False):
    
        keys, values =zip(*dict_.items())
        dicts = [dict(zip(keys, x)) for x in product(*values)]

        if train:
            for idx, dict_ in enumerate(dicts):
                dict_.update({'gpu_id':
                    self.GPU_LIST[(idx // self.model_per_gpu) 
                        % len(self.GPU_LIST)]
                    })

        return dicts


    def _batched_train_test(self):

        smm = SharedMemoryManager()
        smm.start()

        X_shared     = _to_shared_mem(smm, self.X)
        w_shared     = _to_shared_mem(smm, self.w)
        delta_shared = _to_shared_mem(smm, self.delta)

        with Manager() as manager:
            param_dict_mngd =     manager.dict({
                'param_dicts_train': self.param_dicts_train,
                'param_dicts_test':  self.param_dicts_test,
                'data_idx':          self.data_idx,
                'X_shared_name':     X_shared.name,
                'X_shape':           self.X.shape,
                'X_dtype':           self.X.dtype,
                'w_shared_name':     w_shared.name,
                'w_shape':           self.w.shape,
                'w_dtype':           self.w.dtype,
                'delta_shared_name': delta_shared.name,
                'delta_shape':       self.delta.shape,
                'delta_dtype':       self.delta.dtype,
                'batch_size':        self.batch_size,
                'batch_idx':         None,
                'test_block_size':   self.test_block_size
                        })

            rslt_mngd =          manager.list([0]*len(self.param_dicts_test))

            for batch_idx in tqdm(range(int(np.ceil(len(self.param_dicts_train)/self.batch_size))),
                    desc="batched cross validation"):

                param_dict_mngd['batch_idx'] = batch_idx

                '''
                _run_batch_process(param_dict_mngd, rslt_mngd)
                print (rslt_mngd)
                raise
                '''

                p = mp.Process(target = _run_batch_process, args = (param_dict_mngd, rslt_mngd))

                p.start() 
                p.join()
                
                p.terminate()
                if p.exitcode != 0:
                    raise MemoryError("ERROR: Cross validation did not finish successfully due to memory error. Consider reducing nthread and/or models_per_gpu to cut down on the memory used by cross validation.")
            
            smm.shutdown()
            
            self.rslts = list(rslt_mngd)

    def _calculate_output_statistics(self):
        def sort_pred(dict_):
            return (dict_['max_depth'], dict_['n_estimators'], dict_['fold'])

        #self.param_dicts_test = list(self.param_dicts_test)
        #self.rslts            = list(self.rslts)

        rslt__param_dict_test = \
            [(rslt, param_dict_test) for rslt, param_dict_test in \
                sorted(zip(self.rslts,self.param_dicts_test), 
                key=lambda pair: sort_pred(pair[1]))
                    ]
        srtd_rslts, srtd_param_dict_test = zip(*rslt__param_dict_test)
        srtd_rslts = np.array(srtd_rslts)
        srtd_rslts = srtd_rslts.reshape(-1, len(self.cv))

        self.srtd_param_dict_test_scores = srtd_rslts.mean(axis=1)
        self.srtd_param_dict_test_std    = srtd_rslts.std(axis=1)
 
        self.srtd_param_dict_test = srtd_param_dict_test[0::len(self.cv)]
        for srtd_param_dict_test in self.srtd_param_dict_test:
            srtd_param_dict_test.pop('fold')


    def fit(self, X, w, delta):
        self.X     = np.array(X)
        self.w     = np.array(w)
        self.delta = np.array(delta)

        self._batched_train_test()

        self._calculate_output_statistics()


        return {
            "params":          np.array(self.srtd_param_dict_test, dtype='object'),
            "mean_test_score": self.srtd_param_dict_test_scores,
            "std_test_score":  self.srtd_param_dict_test_std,
            "se_test_score":   self.srtd_param_dict_test_std/np.sqrt(len(self.cv)),
            "best_params":     self.srtd_param_dict_test[np.argmax(self.srtd_param_dict_test_scores)]
        }


def cv(param_grid, X_post, num_folds, gpu_list, nthread=-1, models_per_gpu=1):
    """cross validate different hyperpatameter combinations based on likelihood risk.

    :param param_grid: a dictionary containing candidate values for the hyperparameters to cross-validate on. An example would be:
    param_grid = {
        'max_depth':    [1, 2, 3, 4, 5],
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'eta':          [0.1]
    }
    :type param_grid: dict
    :param X: he preprocessed covariate matrix. An output of the preprocessor.
    :type X: pd.DataFrame
    :param w: length of each epoch.  An output of the preprocessor.
    :type w: pd.Series
    :param delta: an indicator which equals one if event occurred at the end of epoch; zero otherwise. An output of the preprocessor.
    :type delta: pd.Series
    :param ID: subject ID of each row in \textbf{X}, \textbf{w}, and \textbf{delta}.  An output of the preprocessor.
    :type ID: pd.Series
    :param num_folds: the number $K$ in $K$-fold cross-validation.
    :type num_folds: int
    :param gpu_list: the list of GPU IDs to use for training. Set $\text{gpu\_list} = [-1]$ to use CPUs.
    :type gpu_list: list
    :param nthread: 
    :type batch_size: int
    :param models_per_gpu: 
    :type batch_size: int
    :return: \textbf{cv\_rslts}: mean and st.dev of the log-likelihood value for each hyperparameter combination.
    :rtype: dict
    """

    if gpu_list == [-1]:
        batch_size = nthread if nthread != -1 else mp.cpu_count()-1
    else:
        batch_size = models_per_gpu * len(gpu_list)

    X, w, delta, ID = indexable(X_post['X'], X_post['w'], X_post['delta'], X_post['ID'])

    gkf = list(GroupKFold(n_splits=num_folds).split(X,delta,ID))
    
    ID             = ID.astype(int)
    ID_unique_srtd = np.sort(np.unique(ID))

    nfolds = num_folds

    ID_counts = [len(ID_unique_srtd)//nfolds] * nfolds
    for i in range(len(ID_unique_srtd)%nfolds):
        ID_counts[i]+=1

    assert sum(ID_counts)==len(ID_unique_srtd)

    fold_idxs = np.hstack([[i]*id_count for i, id_count in enumerate(ID_counts)])

    #np.random.seed(666)
    np.random.shuffle(fold_idxs)

    gkf_ = []
    for fold_idx in range(nfolds):
        train_ids = ID_unique_srtd[np.where(fold_idxs!=fold_idx)[0]]
        test_ids  = ID_unique_srtd[np.where(fold_idxs==fold_idx)[0]]
        gkf_.append((np.where(np.isin(ID, train_ids))[0], np.where(np.isin(ID, test_ids))[0]))

    gkf = gkf_

    collapsed_ntree_gs_  = collapsed_gs_(param_grid, gkf, gpu_list, batch_size)
 
    try:
        results     = collapsed_ntree_gs_.fit(X,w,delta)
        means       = results['mean_test_score']
        stds        = results['std_test_score']
        params      = results['params']

    except MemoryError as me:
        print (me)
        means       = np.array([])
        stds        = np.array([])
        params      = np.array([])


    return {"params":params , "score_mean":means, "score_ste":stds/np.sqrt(num_folds)}


def best_param_1se_rule(cv_results, model_complexity, bounded_search=True):
    params, means, stes               = [cv_results[key] for key in ["params", "score_mean", "score_ste"]]
    highest_mean_idx                  = np.argmax(means)
    highest_mean, highest_mean_ste    = means[highest_mean_idx], stes[highest_mean_idx]
    params_within_1se                 = [param for (param, mean) in zip(params, means) if abs(mean-highest_mean)<highest_mean_ste]
    params_within_1se                 = [{key: params_[key] for key in ['max_depth', 'n_estimators']} for params_ in params_within_1se]
    if bounded_search:
        best_params                   = params[highest_mean_idx]
        params_within_1se             = [params for params in params_within_1se
                                      if all([params[key]<=best_params[key] for key in ['max_depth', 'n_estimators']])]
    #if model_complexity is None:
    #    model_complexity              = lambda max_depth, n_estimators: n_estimators*(2**max_depth)
    return min(params_within_1se, key = lambda params:model_complexity(params['max_depth'], params['n_estimators'])), params_within_1se