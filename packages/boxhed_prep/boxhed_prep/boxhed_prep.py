import os
import sys
import copy
import glob
import shutil
import pathlib
import warnings
import subprocess
import numpy as np
import pandas as pd
from ctypes import *

def build(curr_dir: str):
    """build the shared library of preprocessor. Runs cmake under the hood.

    :param curr_dir: current working directory in which CMakeLists.txt exists.
    :type curr_dir: str
    """
    def cmake_args():
        if sys.platform == 'win32':
            return ["-GVisual Studio 17 2022", "-A x64"]
        return []

    def cmake_build_args():
        
        if sys.platform == 'win32':
            return  ['--config Release']
        return []

    build_dir = os.path.join(curr_dir, './build')

    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.mkdir(build_dir)
    subprocess.run(["cmake", '..']           + cmake_args(),       cwd = build_dir, stdout=subprocess.DEVNULL)#, shell=True)
    subprocess.run(["cmake", '--build', '.'] + cmake_build_args(), cwd = build_dir, stdout=subprocess.DEVNULL)#, shell=True)


def get_prep_lib_dir():
    """find the built shared library if exists, build it otherwise.
    """
    def lib_name():
        if sys.platform == 'win32':
            return '_boxhed_prep.dll'
        return 'lib_boxhed_prep.so'

    curr_dir = pathlib.Path(__file__).parent.resolve()
    lib_addr_ptrn = os.path.join(curr_dir, f'build{os.sep}**{os.sep}{lib_name()}')

    try:
        return glob.glob(lib_addr_ptrn, recursive=True)[0]
    except IndexError as e:
        warnings.warn("BoXHED Preprocessor could not find compiled C++ module. Building...", RuntimeWarning)
        build(curr_dir)
        return glob.glob(lib_addr_ptrn, recursive=True)[0]

class preprocessor:

    class c_boundary_info (Structure):
        _fields_ = [("nIDs", c_size_t), 
                    ("out_nrows", c_size_t), 
                    ("in_lbs",    c_void_p),
                    ("out_lbs",   c_void_p)]

    def __init__(self):

        self.prep_libfile = get_prep_lib_dir()

        self.prep_lib = cdll.LoadLibrary(self.prep_libfile)

        # initializing functions to be used as an interface between Python and C++ in ctypes.
        self.prep_lib.compute_quant.restype    = None
        self.prep_lib.compute_quant.argtypes   = [
                c_void_p, #data_v
                c_size_t, #nrows 
                c_size_t, #ncols
                c_void_p, #is_cat_v
                c_size_t, #t_start_idx
                c_size_t, #t_end_idx
                c_size_t, #id_idx
                c_size_t, #delta_idx
                c_void_p, #quant_v
                c_void_p, #quant_size_v
                c_size_t, #num_quantiles
                c_bool,   #weighted
                c_int     #nthread
                ]

        self.prep_lib.get_boundaries.restype   = c_void_p
        self.prep_lib.get_boundaries.argtypes  = [
                c_void_p, #data_v
                c_size_t, #nrows
                c_size_t, #ncols
                c_size_t, #nIDs
                c_size_t, #id_col_idx
                c_size_t, #t_start_idx
                c_size_t, #t_end_idx
                c_void_p, #quant_v
                c_void_p, #quant_size_v
                c_size_t  #num_quantiles
                ]

        self.prep_lib.preprocess.restype       = None
        self.prep_lib.preprocess.argtypes      = [
                c_void_p, #data_v
                c_size_t, #nrows
                c_size_t, #ncols
                c_void_p, #is_cat_v
                c_void_p, #bndry_info
                c_void_p, #out_data_v
                c_void_p, #quant_v
                c_void_p, #quant_size_v
                c_size_t, #num_quantiles
                c_size_t, #t_start_idx 
                c_size_t, #t_end_idx
                c_size_t, #delta_idx
                c_size_t, #id_col_idx
                c_int     #nthread 
                ]

        self.prep_lib.free_boundary_info.restype  = None
        self.prep_lib.free_boundary_info.argtypes = [
                c_void_p #bndry_info
                ]

        self.prep_lib.shift_left.restype  = None
        self.prep_lib.shift_left.argtypes = [
                c_void_p, #data_v 
                c_size_t, #nrows
                c_size_t, #ncols
                c_void_p, #quant_idx_v
                c_void_p, #quant_v
                c_void_p, #quant_size_v
                c_size_t, #num_quantiles
                c_int     #nthread
                ]
 

    def _set_col_indcs(self):
        """find the indexes of non-covariate columns. These should be present in the data.
        """
        self.t_start_idx = self.colnames.index('t_start')
        self.id_idx      = self.colnames.index('ID')
        self.t_end_idx   = self.colnames.index('t_end')
        self.delta_idx   = self.colnames.index('delta') #TODO: either 0 or 1

    def _contig_float(self, arr):
        return np.ascontiguousarray(arr, dtype = np.float64)

    def _contig_size_t(self, arr):
        return np.ascontiguousarray(arr, dtype = np.uintp)

    def _contig_bool(self, arr):
        return np.ascontiguousarray(arr, dtype = np.bool_)

    def __compute_quant(self, data, nrows, ncols, is_cat):
        self.prep_lib.compute_quant(
            c_void_p(data.ctypes.data), 
            c_size_t(nrows), 
            c_size_t(ncols), 
            c_void_p(is_cat.ctypes.data),
            c_size_t(self.t_start_idx), 
            c_size_t(self.t_end_idx), 
            c_size_t(self.id_idx), 
            c_size_t(self.delta_idx), 
            c_void_p(self.quant.ctypes.data), 
            c_void_p(self.quant_size.ctypes.data),
            c_size_t(self.num_quantiles),
            c_bool(self.weighted),
            c_int(self.nthread))


    def _get_boundaries(self, data, nrows, ncols, nIDs):
        return self.c_boundary_info.from_address(self.prep_lib.get_boundaries(
            c_void_p(data.ctypes.data), 
            c_size_t(nrows), 
            c_size_t(ncols), 
            c_size_t(nIDs), 
            c_size_t(self.id_idx), 
            c_size_t(self.t_start_idx), 
            c_size_t(self.t_end_idx), 
            c_void_p(self.quant.ctypes.data), 
            c_void_p(self.quant_size.ctypes.data),
            c_size_t(self.num_quantiles)
            ))

    def _preprocess(self, data, nrows, ncols, is_cat, bndry_info):
        preprocessed = self._contig_float(np.zeros((bndry_info.out_nrows, ncols))) 

        self.prep_lib.preprocess(
                c_void_p(data.ctypes.data),
                c_size_t(nrows), 
                c_size_t(ncols), 
                c_void_p(is_cat.ctypes.data),
                byref(bndry_info), 
                c_void_p(preprocessed.ctypes.data),
                c_void_p(self.quant.ctypes.data), 
                c_void_p(self.quant_size.ctypes.data),
                c_size_t(self.num_quantiles), 
                c_size_t(self.t_start_idx), 
                c_size_t(self.t_end_idx), 
                c_size_t(self.delta_idx), 
                c_size_t(self.id_idx), 
                c_int(self.nthread))

        return preprocessed

    def _free_boundary_info(self, bndry_info):
        """free the array of boundary_info_ (defined in boxhed_prep.h, created in boxhed_prep.cpp).

        :param bndry_info: the array of boundary_info_ (defined in boxhed_prep.h, created in boxhed_prep.cpp)
        :type bndry_info: c_void_p
        """
        self.prep_lib.free_boundary_info(byref(bndry_info))
        del bndry_info

    def _prep_output_df(self, preprocessed):
        """prepare the preprocessed dataset as the output. Change column names if necessary, divide into id, X, delta, and w.

        :param preprocessed: preprocessed Pandas dataframe.
        :type preprocessed: pd.DataFrame
        :return: preprocessed data in 4 separate parts: pd.Series, pd.DataFrame, pd.Series, pd.Series
        :rtype: _type_
        """
        new_col_names                  = copy.copy(self.colnames)
        new_col_names[self.t_end_idx]  = 'dt'

        preprocessed = pd.DataFrame(preprocessed, columns = new_col_names)
        id           = preprocessed['ID']
        #self.y           = self.preprocessed[['delta', 'dt']]
        w            = preprocessed['dt']
        delta        = preprocessed['delta']
        X            = preprocessed.drop(columns = ['ID', 'delta', 'dt'])

        return id, X, delta, w
    

    def _data_sanity_check(self, data):
        assert data.ndim==2,"ERROR: data needs to be 2 dimensional"
        #assert data['subject'].between(1, nIDs).all(),"ERROR: Patients need to be numbered from 1 to # subjects"

    def _setup_data(self, data):

        nIDs = data['ID'].nunique()

        self._data_sanity_check(data)
        data  = self._contig_float(data)

        return data, nIDs


    def _compute_quant(self, data, nrows, ncols, is_cat, split_vals):
        """compute quantiles to be used for preprocessing.

        :param data: input data before preprocessing.
        :type data: pd.DataFrame
        :param nrows: input data number of rows.
        :type nrows: int
        :param ncols: input data number of columns.
        :type ncols: int
        :param is_cat: a list of indexes of columns that contain one-hot encoded columns.
        :type is_cat: list
        :param split_vals: a dictionary manually specifying the candidate splits for each column.
        :type split_vals: dict
        """
        for k, _ in split_vals.items():
            if k in ["ID", "delta"]:
                raise ValueError(f'The column {k} was passed in split_vals but is not used as feature to split on.')

            if k in ["t_start", "t_end"]:
                raise ValueError(f'The column {k} was passed in split_vals. If specifying time splits, use the key "time" in split_vals instead of {k}.')

        split_vals = {(k if k!="t" else "t_start"):v for (k,v) in split_vals.items()}
        #split_vals = {k:np.sort(np.unique(v)) for (k,v) in split_vals.items()}
        
        max_nsplits = max([len(v) for (k,v) in split_vals.items()], default=0)
        if max_nsplits > self.num_quantiles:
            raise ValueError(f'The number of unique split values specified in split_vals for any covariate/time ({max_nsplits}) cannot exceed number of quantiles num_quantiles ({self.num_quantiles}).')

        self.quant      = self._contig_float(np.zeros((1, self.num_quantiles*(ncols))))
        self.quant_size = self._contig_size_t(np.zeros((1, ncols)))

        self.__compute_quant(data, nrows, ncols, is_cat)

        for k, v in split_vals.items():
            try:
                idx = self.colnames.index(k)
            except ValueError:
                raise ValueError(f'The column {k} was passed in split_vals but does not exist in the dataset.')

            v = np.append(v, 0 if idx == self.t_start_idx else data[:,idx].min()-1)
            v = np.sort(np.unique(v))
            assert len(v) <= self.num_quantiles, "The specified split_vals values is not compatible with num_quantiles. Consider increasing num_quantiles by at least 1."

            self.quant[0, idx*self.num_quantiles:idx*self.num_quantiles+len(v)] = v
            self.quant_size[0, idx] = len(v)


    def preprocess(self, data, is_cat=[], split_vals={}, num_quantiles=256, weighted=False, nthread=1):
        """preprocess the input data.

        :param data: training data.
        :type data: pd.DataFrame
        :param is_cat: a list of the column indexes that contain categorical data. The categorical data must be one-hot encoded. For example, is\_cat = [4,5,6] if a categorical variable with 3 factors is transformed into binary-valued columns 4,5,6., defaults to []
        :type is_cat: list, optional
        :param split_vals: a dictionary to specify values to split on for any covariate or time. The key should the variable name and the value a list (or a 1d NumPy array) containing candidate split points. For specifying candidate points for time the key value should simply be 'time' and for other covariates it should exactly match the column name in the dataset. Key values 'ID', 't_start', 't_end', and 'delta' are not allowed. If the candidate split points of a covariate/time is not specified in split_vals, extracted quantiles (the default behavior) would be used. Be sure to not specify split points whose number exceeds the num_quantiles variable., defaults to {}
        :type split_vals: dict, optional
        :param num_quantiles: the number of candidate split points to try for time and for each covariate. The locations of the split points are based on the quantiles of the training data., defaults to 256
        :type num_quantiles: int, optional
        :param weighted: if set to True, the locations of the candidate split points will be based on weighted quantiles, defaults to False
        :type weighted: bool, optional
        :param nthread: number of CPU threads to use for preprocessing the data, defaults to 1
        :type nthread: int, optional
        :return:
            \textbf{ID}: subject ID for each row in the processed data frames X, w, and delta.
                
            \textbf{X}: each row represents an epoch of the transformed data, and contains the values of the covariates as well as its start time.
                
            \textbf{w}: length of each epoch.
                
            \textbf{delta}: equals one if an event occurred at the end of the epoch; zero otherwise.
        :rtype: pd.Series, pd.DataFrame, pd.Series, pd.Series
        """

        #XXX: using np.float64---c_double
        self.nthread            = nthread
        self.num_quantiles      = num_quantiles#min(num_quantiles, 256)
        self.weighted           = weighted
        _is_cat                 = self._contig_bool(np.zeros((1, data.shape[1])))
        for cat_col in is_cat:
            _is_cat[0, cat_col] = True
        self.is_cat             = _is_cat
        nrows                   = data.shape[0]
        self.ncols              = data.shape[1]

        self.colnames  = list(data.columns)
        self._set_col_indcs()

        data, nIDs              = self._setup_data(data)

        self._compute_quant(data, nrows, self.ncols, _is_cat, split_vals)

        bndry_info              = self._get_boundaries(data, nrows, self.ncols, nIDs)
        preprocessed            = self._preprocess(data, nrows, self.ncols, _is_cat, bndry_info)
        ID, X, delta, w         = self._prep_output_df(preprocessed)
        self._free_boundary_info(bndry_info)

        return {"ID": ID, "X": X, "w": w, "delta": delta}

    def _post_training_get_X_shape(self, X):
        assert X.ndim==2,"ERROR: data needs to be 2 dimensional"
        nrows, ncols = X.shape
        assert ncols == self.ncols-3, "ERROR: ncols in X does not match the trained data"
        return nrows, ncols

    def shift_left(self, X):

        nrows, ncols = self._post_training_get_X_shape(X)

        quant_idxs = np.ascontiguousarray(np.zeros(ncols), dtype = np.int32)

        for idx, colname in enumerate(X.columns):
            if colname in ['t', 'time', 't_start']:
                colname = 't_start'
            col_idx = self.colnames.index(colname)
            assert col_idx > -1, "ERROR: X and trained data colnames do not match"
            quant_idxs [idx] = col_idx


        processed = np.ascontiguousarray(X.values)

        self.prep_lib.shift_left(
            c_void_p(processed.ctypes.data),
            c_size_t(nrows),
            c_size_t(ncols),
            c_void_p(quant_idxs.ctypes.data),
            c_void_p(self.quant.ctypes.data),
            c_void_p(self.quant_size.ctypes.data),
            c_size_t(self.num_quantiles),
            c_int(self.nthread))
        
        return processed


    def update_time_splits(self, time_splits):
        idx              = self.t_start_idx
        time_splits      = np.sort(np.unique(np.append(
                           time_splits, self.quant[0,idx* self.num_quantiles]
                           )))
        time_splits_size = time_splits.shape[0]

        self.quant[0,   (idx  ) * self.num_quantiles :
                        (idx+1) * self.num_quantiles ]                   = 0

        self.quant[0,   (idx  ) * self.num_quantiles :
                        (idx  ) * self.num_quantiles + time_splits_size] = time_splits

        self.quant_size[0, idx] = time_splits_size


    def epoch_break_cte_hazard (self, data): # used for breaking epochs into cte hazard valued intervals
        nrows, ncols  = data.shape
        data, nIDs    = self._setup_data(data)
        bndry_info    = self._get_boundaries(data, nrows, ncols, nIDs)
        data          = self._preprocess(data, nrows, ncols, self.is_cat, bndry_info)
        
        processed       = pd.DataFrame(data, columns = self.colnames)
        processed.rename(columns={"t_end": "dt"}, inplace=True)

        self._free_boundary_info(bndry_info)
        
        return processed