'''
Generate appropriate plots for report
'''
from read_LL_textfile import read_LL_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = "mpi_results/10nodes_10steps_80grid/T_2.0/LL-Output-Mon-06-Nov-2023-at-11-40-26AM-80-2.0.txt"
data = read_LL_file(file)


fig,ax = plt.subplots(1,2)
index = np.arange(len(data.index))
ax[0].scatter(index,data["Energy"], label = "energy")
ax[1].scatter(index,data["Order"], color="orange", label ="order")
plt.legend()
plt.show()