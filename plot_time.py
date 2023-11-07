import numpy as np
import matplotlib.pyplot as plt
nparr = np.loadtxt("np.csv",delimiter=",", dtype=float)
numbaarr=np.loadtxt("numba.csv",delimiter=",", dtype=float)
cythonarr=np.loadtxt("cython.csv",delimiter=",", dtype=float)
mpiarr=np.loadtxt("mpi.csv",delimiter=",", dtype=float)

main =[13.780280,54.549492,132.769379,311.576499]
gridsize = [20,40,60,80]
fig, ax = plt.subplots(1,1)
ax.plot(gridsize,nparr, marker='o',color = "orange", label = "Numpy")

ax.plot(gridsize,cythonarr,marker='o', color = "red", label = "Cython")

ax.plot(gridsize,mpiarr, marker='o',color = "pink", label = "MPI")

ax.plot(gridsize,numbaarr,marker='o', color = "purple", label = "Numba")

ax.plot(gridsize,main,marker='o', color = "blue", label = "Serial")
#
plt.legend()
ax.set_xlabel("Lattice Size, N")
ax.set_ylabel("Time to Run 1000 MC Steps, s ")
plt.savefig("timeplot.png")