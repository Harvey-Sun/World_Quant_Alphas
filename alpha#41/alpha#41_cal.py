# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:05:02 2017

@author: LZJF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('C:\cStrategy\Factor')


def import_data(name):
    data = pd.read_csv('%s.csv' % name, index_col=0)
    data.index = pd.to_datetime([str(i) for i in data.index])
    return data

high = import_data('LZ_GPA_QUOTE_THIGH')
low = import_data('LZ_GPA_QUOTE_TLOW')
close = import_data('LZ_GPA_QUOTE_TCLOSE')
value = import_data('LZ_GPA_QUOTE_TVALUE')
volume = import_data('LZ_GPA_QUOTE_TVOLUME')
stop = import_data('LZ_GPA_SLCIND_STOP_FLAG')

factor41 = ((high * low) ** 0.5 - (value / volume) * 10) / close


factor_name = 'factor41_1D'

factor41.index.name = factor_name
os.chdir('F:\Factors\World_Quant_Alphas\#41')
factor41.to_csv('%s.csv' % factor_name)
