from datetime import datetime
from pytz import timezone
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from py3nvml import get_free_gpus
import pickle
from joblib import Parallel, delayed
import itertools
from collections import namedtuple
from scipy.stats import beta # beta distribution.
import math


import warnings
warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd
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

import functools
import multiprocessing
from sys import platform
if platform == "linux" or platform == "linux2":
    multiprocessing.set_start_method('fork')
from multiprocessing import Process, Queue


def run_as_process(func):
    @functools.wraps(func)
    def run_as_process(*args, **kwargs):

        def _func(queue, func, *args, **kwargs):
            queue.put(func(*args, **kwargs))

        queue = Queue()
        p = Process(target=_func, args=(queue, func, *args), kwargs=kwargs)
        p.start()
        p.join()
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


