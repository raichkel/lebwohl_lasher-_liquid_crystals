'''
Generate appropriate plots for report
'''
from read_LL_textfile import read_LL_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = "mpi_results/10nodes_10steps_20grid/T_0.2/LL-Output-Thu-26-Oct-2023-at-11-43-08AM-20-0.2.txt"
data = read_LL_file(file)


fig,ax = plt.subplots(1,2)
index = np.arange(len(data.index))
ax[0].scatter(index,data["Energy"])
ax[1].scatter(index,data["Order"])

plt.show()