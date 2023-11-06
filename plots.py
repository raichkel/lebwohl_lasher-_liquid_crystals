'''
Generate avg S plot for report. Find where energy equilibriates.
'''
from read_LL_textfile import read_LL_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

folder = "cython_results"
extension = ".txt"

def slope(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)
# check it has equilibriated
def equilib(file):
    data = read_LL_file(file)
    # fig,ax0 = plt.subplots(1,1)
    # index = np.arange(len(data.index))
    # ax0.scatter(index,data["Energy"], color = "purple")
    # ax0.set_xlabel("MC Step")
    # ax0.set_ylabel("Reduced Energy U/$\epsilon$")
    # plt.tight_layout()
    # plt.savefig("energy.png")
    range_arr = np.arange(0,1000,10)
    first = False
    for i in range(len(range_arr)-1):
        if first == False:
            x1 = range_arr[i]
            y1 = data["Energy"].iloc[i]
            i+=1
            x2 = range_arr[i]
            y2 = data["Energy"].iloc[i]
            s = slope(x1,y1,x2,y2)
            # we will (based on small data sample, and from observation), define 'equilibriated' as
            # -20 <= gradient <=20
            
            if s>-21 and s<21:
                #print(f"equilibrium at : {x2}")
                first = True


    # fig,ax1 = plt.subplots(1,1)
    # ax1.scatter(index,data["Order"], color="orange")
    # ax1.set_xlabel("MC Step")
    # ax1.set_ylabel("Order Parameter S")
    # plt.tight_layout()
    # plt.savefig("order.png")


    # look at averaging S from x2 onwards
    avg_order = data["Order"].iloc[60:].mean()
    return avg_order


for nfolder in os.listdir(folder):
    # looping over folders 20_grid, 40_grid etc etc
    nmax = nfolder.split("_")
    nmax = int(nmax[0])
    # creating one t arr and one s arr per grid
    temp_arr = np.zeros(10)
    order_arr = np.zeros(10)
    i = 0
    for Tfolders in os.listdir(os.path.join(folder,nfolder)):
        # in folder 20_grid, looping over T_0.2, T_0.4 etc etc
        #print(i)
        
        part = Tfolders.split("_")
        temp_arr[i] = float(part[1])
        # get temp from folder name
        
        for file in os.listdir(os.path.join(folder,nfolder,Tfolders)):
            # getting LL.txt file in each T folder
            if file.endswith(extension):
                
                order_val = equilib(os.path.join(folder,nfolder,Tfolders,file))
                order_arr[i] = order_val
        i+=1

    # print(temp_arr) 
    # print(order_arr)
    fig, ax = plt.subplots(1,1)    
    ax.plot(temp_arr, order_arr, color = "pink")
    ax.scatter(temp_arr, order_arr, color = "magenta")
    ax.set_xlabel("T*")
    ax.set_ylabel("Order S")
    plt.savefig(f"S_T_plot_{nmax}.png")
    plt.clf()
        


        



