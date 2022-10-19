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

def build(curr_dir):

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
    def lib_name():
        if sys.platform == 'win32':
            return '_boxhed_prep.dll'
        #elif sys.platform.startswith('linux') or sys.platform.startswith('freebsd'):
        return 'lib_boxhed_prep.so'
        #else:
        #    raise OSError("ERROR: platform not supported!")

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
                c_size_t, #num_quantiles
                c_int     #nthread
                ]
 

    def _set_col_indcs(self):
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
        self.prep_lib.free_boundary_info(byref(bndry_info))
        del bndry_info

    def _prep_output_df(self, preprocessed):
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

        #making sure subject data is contiguous
        #data = data.sort_values(by=['ID', 't_start'])
        nIDs = data['ID'].nunique()

        self._data_sanity_check(data)
        data  = self._contig_float(data)

        return data, nIDs


    def _compute_quant(self, data, nrows, ncols, is_cat, split_vals):
        for k, _ in split_vals.items():
            if k in ["ID", "delta"]:
                raise ValueError(f'The column {k} was passed in split_vals but is not used as feature to split on.')

            if k in ["t_start", "t_end"]:
                raise ValueError(f'The column {k} was passed in split_vals. If specifying time splits, use the key "time" in split_vals instead of {k}.')

        split_vals = {(k if k!="t" else "t_start"):v for (k,v) in split_vals.items()}
        split_vals = {k:np.sort(np.unique(v)) for (k,v) in split_vals.items()}
        
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

            col_min  = data[:,idx].min()
            
            if  v[0] < col_min:
                v[0] = col_min

            if  v[0] > col_min:
                v    = np.sort(np.unique(np.append(v, col_min)))
                assert len(v) <= self.num_quantiles, "The specified split_vals values is not compatible with num_quantiles. Consider increasing num_quantiles by at least 1."

            self.quant[0, idx*self.num_quantiles:idx*self.num_quantiles+len(v)] = v
            self.quant_size[0, idx] = len(v)


    def preprocess(self, data, is_cat=[], split_vals={}, num_quantiles=256, weighted=False, nthread=1):
        #TODO: maye change how the data is given? pat, X, y?

        #XXX: using np.float64---c_double
        self.nthread            = nthread
        self.num_quantiles      = num_quantiles#min(num_quantiles, 256)
        self.weighted           = weighted
        _is_cat                 = self._contig_bool(np.zeros((1, data.shape[1])))
        for cat_col in is_cat:
            _is_cat[0, cat_col] = True
        self.is_cat             = _is_cat
        nrows                   = data.shape[0]
        ncols                   = data.shape[1]

        self.colnames  = list(data.columns)
        self._set_col_indcs()

        data, nIDs              = self._setup_data(data)

        self._compute_quant(data, nrows, ncols, _is_cat, split_vals)

        bndry_info              = self._get_boundaries(data, nrows, ncols, nIDs)
        preprocessed            = self._preprocess(data, nrows, ncols, _is_cat, bndry_info)
        IDs, X, delta, w        = self._prep_output_df(preprocessed)
        self._free_boundary_info(bndry_info)

        return IDs, X, w, delta

    def _post_training_get_X_shape(self, X):
        assert X.ndim==2,"ERROR: data needs to be 2 dimensional"
        nrows, ncols = X.shape
        assert ncols == self.ncols-3, "ERROR: ncols in X does not match the trained data"
        return nrows, ncols

    def shift_left(self, X):

        nrows, ncols = self._post_training_get_X_shape(X)

        quant_idxs = np.ascontiguousarray(np.zeros(ncols), dtype = np.int32)

        for idx, colname in enumerate(X.columns):
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
            c_size_t(self.num_quantiles),
            c_int(self.nthread))
        
        return processed

    def epoch_break_cte_hazard (self, data): # used for breaking epochs into cte hazard valued intervals
        nrows, ncols  = data.shape
        data, nIDs    = self._setup_data(data)
        bndry_info    = self._get_boundaries(data, nrows, ncols, nIDs)
        data          = self._preprocess(data, nrows, ncols, self.is_cat, bndry_info)
        
        processed       = pd.DataFrame(data, columns = self.colnames)
        processed.rename(columns={"t_end": "dt"}, inplace=True)

        self._free_boundary_info(bndry_info)
        
        return processed