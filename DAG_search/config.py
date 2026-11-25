import json
import os
from pathlib import Path

import pandas as pd

FEATURE_MAPPING_PATH = '/public/home/jiangxx/a/main_index_new@20260204/mappings.csv'
DEFAULT_FUNDAMENTAL_DIR = '~/stock_dataset/new_tabular_data'
DEFAULT_FEATURE_SAMPLE_FILE = '20250730.csv'
FEATURE_DICT_PATH = './UDFS/feature_dict.json'

'''
Numeric Operations
'''
NODE_ARITY = {
    '+' : 2,
    '*' : 2,
    '=' : 1,
    'inv' : 1,
    #'neg' : 1,
    #'sin' : 1, 
    #'cos' : 1,
    'exp' : 1,
    'log' : 1,
    'sqrt' : 1,
    'mean': 1,
    'std' : 1,
    'ref' : 1,
    'sum' : 1,
    'var' : 1,
    'max' : 1,
    'min' : 1,
    'greater' : 2,
    'less' : 2,
    #'sq' : 1,
    'sub_l' : 2,
    'sub_r' : 2,
    'div_l' : 2,
    'div_r' : 2,
    #'corr': 2
    #'pow_l' : 2,
    #'pow_r' : 2,
}

NODE_STR = {
    '+' : '(a)+(b)',
    '*' : '(a)*(b)',
    '=' : '(a)',
    'inv' : '1/(a)',
    #'neg' : '-a',
    #'sin' : 'sin(a)',
    #'cos' : 'cos(a)',
    'exp' : 'Exp((a))',
    'log' : 'Log((a))',
    'sqrt' : 'Sqrt((a))',
    #'sq' : '(a)**2',
    'sub_l' : '(a)-(b)',
    'sub_r' : '(b)-(a)',
    'div_l' : '((a))/((b))',
    'div_r' : '((b))/((a))',
    'mean': 'mean((a))',
    'std' : 'std((a))',
    'ref' : 'ref((a))',
    'sum' : 'sum((a))',
    'var' : 'var((a))',
    'max' : 'max((a))',
    'min' : 'min((a))',
    'greater' : 'Greater((a),(b))',
    'less' : 'Less((a),(b))',
    #'corr' : 'corr(a,b)'
    #'pow_l' : '(a)**(b)',
    #'pow_r' : '(b)**(a)',
}

'''
Symbolic Operations
'''

NODE_ID = {
    '+' : 1,
    '*' : 2,
    '=' : 3,
    'inv' : 4,
    #'neg' : 7,
    #'sin' : 5,
    #'cos' : 6,
    'exp' : 5,
    'log' : 6,
    'sqrt' : 7,
    #'sq' : 13,
    'sub_l' : 8,
    'sub_r' : 9,
    'div_l' : 10,
    'div_r' : 11,
    'mean': 12,
    'ref': 13,
    'std': 14,
    'sum': 15,
    'var': 16,
    'max': 17,
    'min': 18,
    'greater': 19,
    'less': 20,
    #'corr': 17
    #'pow_l' : 18,
    #'pow_r' : 19,
}

ROLLING_DICT = {
    "mean" : "Mean",
    "std" : "Std",
    "ref" : "Ref",
    "max" : "Max",
    "var" : "Var",
    "min" : "Min",
    "sum" : "Sum",
    #"corr" : "Corr"
}

feature_dict = {
    'x_0': 'open',
    'x_1': 'close',
    'x_2': 'high',
    'x_3': 'low',
    'x_4': 'volume',
}