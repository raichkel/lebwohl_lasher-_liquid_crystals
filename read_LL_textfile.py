'''
dummy code to read in values from the produced LL.txt files
'''
import pandas as pd
import numpy as np

filename = "tests/10steps_20grid/LL-Output-Thu-26-Oct-2023-at-10-41-18AM.txt"

data = pd.read_csv(filename, header = 7, delim_whitespace=True)

# data written the same. real data will always be from line data.iloc[8] onwards
# header is in line data.iloc[6]

data = data.iloc[1:]

dict = {'#':'MC_step', 'MC':'Ratio','step:':'Energy','Ratio:':'Order','Energy:':'0','Order:':'1'}
data.rename(mapper = dict,axis ='columns', inplace=True)
data.drop(['0','1'], axis =1, inplace=True)
data.set_index('MC_step', inplace=True)

# get certain value from column
#print(data.loc[:,'Order'].iloc[2])