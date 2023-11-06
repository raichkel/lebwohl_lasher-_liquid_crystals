'''
Generate appropriate plots for report
'''
from read_LL_textfile import read_LL_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = "LL-Output-Mon-06-Nov-2023-at-10-35-57AM.txt"
data = read_LL_file(file)


fig,ax = plt.subplots(1,2)
index = np.arange(len(data.index))
ax[0].scatter(index,data["Energy"], label = "energy")
ax[1].scatter(index,data["Order"], label = "order", color="orange")
plt.legend()
plt.show()