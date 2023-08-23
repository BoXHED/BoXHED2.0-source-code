from datetime import datetime
from pytz import timezone
import os
import contextlib
import numpy as np
#from sklearn.model_selection import StratifiedKFold
from py3nvml import get_free_gpus
import pickle
#from joblib import Parallel, delayed
#import itertools
#from collections import namedtuple
from scipy.stats import beta # beta distribution.
import math

import functools
#import multiprocessing
#from sys import platform
#if platform == "linux" or platform == "linux2":
#    multiprocessing.set_start_method('fork', force=True)
#from multiprocessing import Process
import multiprocessing as mp
import traceback
import warnings
warnings.simplefilter(action='ignore', category=Warning)

#import pandas as pd
from pathlib import Path
import sys
sys.path.append(os.path.join(os.path.expanduser("~"), "survival_analysis/BoXHED2.0/xgboost/python-package/"))

CACHE_ADDRESS = './tmp/'

# calc_L2: calculate L2 distance and its 95% confidence interval.
def calc_L2(pred, true):
    L2 = (pred-true)**2
    N = pred.shape[0]
    meanL2_sqr = sum(L2)/N # L2 distance
    sdL2_sqr = math.sqrt(sum((L2-meanL2_sqr)**2)/(N-1))
    meanL2 = math.sqrt(meanL2_sqr)
    return {'point_estimate' : meanL2, 
            'lower_CI'       : meanL2-1.96*sdL2_sqr/2/meanL2/math.sqrt(N),
            'higher_CI'      : meanL2+1.96*sdL2_sqr/2/meanL2/math.sqrt(N)}

#%%
# calculate values of hazard function on testing data.
def TrueHaz(X):
    return beta.pdf(X[:,0], 2, 2)*beta.pdf(X[:,1], 2, 2) 

def curr_dat_time ():
    curr_dt = datetime.now(timezone("US/Central"))
    return curr_dt.strftime("%Y_%m_%d_%H:%M")


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def exec_if_not_cached(func):
    @functools.wraps(func)

    def _exec_if_not_cached(file_name, func, *args, **kwargs):

        file_name_ = file_name+'.pkl'
        file_path = os.path.join(CACHE_ADDRESS, file_name_)

        if Path(file_path).is_file():
            with open( file_path, "rb" ) as file_handle:
                return pickle.load(file_handle)

        else:
            create_dir_if_not_exist(os.path.dirname(file_path))
            output = func(*args, **kwargs)

            with open( file_path, "wb" ) as file_handle:
                pickle.dump(output, file_handle)

            return output


    def _func_args_to_str(func, *args, **kwargs):
        output = func.__name__
        for arg in args:
            output += "__"+str(arg)

        for key, val in kwargs.items():
            output += "__"+str(key)+"_"+str(val)

        return output

    def exec_if_not_cached(*args, **kwargs):

        file_name = _func_args_to_str(func, *args, **kwargs)
        return _exec_if_not_cached(file_name, func, *args, **kwargs)

    return exec_if_not_cached

import json

#TODO: sanity checking the addr
def read_config_json(addr):
    try:
        with open ("config.txt", "r") as config_file:
            config = json.loads(config_file.read())
    except:
        print ("ERROR: config.txt not found!")

    return config


import time
def time_now():
    return time.time()

class timer:

    def __init__(self):
        self.t_start = time_now()

    def get_dur(self):
        return round(time_now()-self.t_start, 3)

def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


#https://stackoverflow.com/questions/63758186/how-to-catch-exceptions-thrown-by-functions-executed-using-multiprocessing-proce
class Process(mp.Process):

    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            #raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def run_as_process(func):
    @functools.wraps(func)
    def run_as_process(*args, **kwargs):

        def _func(queue, func, *args, **kwargs):
            queue.put(func(*args, **kwargs))

        queue = mp.Queue()
        p = Process(target=_func, args=(queue, func, *args), kwargs=kwargs)
        p.start()
        p.join()
        ## https://stackoverflow.com/questions/63758186/how-to-catch-exceptions-thrown-by-functions-executed-using-multiprocessing-proce
        if p.exception:
            raise p.exception
        ##
        p.terminate()
        return queue.get()
    return run_as_process


def _free_gpu_list():
    free_gpus = get_free_gpus()
    nom_free_gpus = free_gpus.count(True)
    free_gpu_list_ = [gpu_id for (gpu_id, gpu_free) in enumerate(free_gpus) if gpu_free == True]
    return free_gpu_list_

def _get_free_gpu_list(nom):
    GPU_LIST = _free_gpu_list()
    if len(GPU_LIST) < nom:
        raise RuntimeError("ERROR: Not enough GPUs available!")

    return GPU_LIST[:nom]


import pickle
def dump_pickle(obj, addr):
    with open(addr, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(addr):
    with open(addr, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


from threading import Thread
def run_as_threads(f, kwargs):

    def f_(rslts, idx, **kwargs):
        rslts [idx] = f(**kwargs)

    rslts = [None]*len(kwargs)
    T     = []
    for idx, kwargs_ in enumerate(kwargs):
        t=Thread(target = f_, args = [rslts, idx], kwargs = kwargs_)
        t.start()
        T.append(t)

    for t in T:
        t.join()

    return rslts

#https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)