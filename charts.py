import numpy as np
import matplotlib.pyplot as plt
import os

folder = "cython_results"
extension = ".txt"

def read_time(name):
    file = open(name) 
    content = file.readlines() 

    time_row = content[5]
    time_row = time_row.split(" ")
    time = float(time_row[11])

    return  time

time_arr_gridsize = np.zeros(4)
gridsize = np.zeros(4)
x=0
for nfolder in os.listdir(folder):
    # looping over folders 20_grid, 40_grid etc etc
    nmax = nfolder.split("_")
    nmax = int(nmax[0])
    gridsize[x]=nmax
    i = 0
    time_arr = np.zeros(10)
    

    for Tfolders in os.listdir(os.path.join(folder,nfolder)):
        # in folder 20_grid, looping over T_0.2, T_0.4 etc etc
        # get temp from folder name
        
        for file in os.listdir(os.path.join(folder,nfolder,Tfolders)):
            # getting LL.txt file in each T folder
            if file.endswith(extension):
                # get time
                time = read_time(os.path.join(folder,nfolder,Tfolders,file))
                time_arr[i]=time
        i+=1
    # get average time
    time_arr_gridsize[x] = np.mean(time_arr)
    x+=1

fig, ax = plt.subplots(1,1)
ax.plot(gridsize,time_arr_gridsize, color = "orange")
ax.scatter(gridsize,time_arr_gridsize, color = "purple")
ax.set_xlabel("Lattice Size, N")
ax.set_ylabel("Time to Run 1000 MC Steps, s ")
plt.savefig("timeplot_cython.png")
#plt.show()