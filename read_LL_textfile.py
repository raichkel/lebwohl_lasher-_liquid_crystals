'''
dummy code to read in values from the produced LL.txt files
'''
import pandas as pd
import numpy as np


def read_LL_file(filename):
    '''
    Reads LL.txt file as pandas df. 
    Inputs:
    - filename: str, path to .txt file
    Returns:
    - data: pandas DataFrame, LL simulation output data 

    Usage:
    get certain value from column:
    >> print(data.loc[:,'Order'].iloc[2])
    '''
    data = pd.read_csv(filename, header = 7, delim_whitespace=True)
    # data written the same. real data will always be from line data.iloc[8] onwards
    # header is in line data.iloc[6]
    data = data.iloc[1:]
    # remap columns and drop spares
    dict = {'#':'MC_step', 'MC':'Ratio','step:':'Energy','Ratio:':'Order','Energy:':'0','Order:':'1'}
    data.rename(mapper = dict,axis ='columns', inplace=True)
    data.drop(['0','1'], axis =1, inplace=True)
    data.set_index('MC_step', inplace=True)

    return data

