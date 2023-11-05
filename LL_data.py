from LebwohlLasher import initdat, one_energy, get_order
import numpy as np
import random
'''
generate test data for pytest uisng LebwohlLasher 
write this to a file for exporting to other branches.
'''
nmax = 20
arr = initdat(nmax)
print(np.shape(arr))
np.savetxt("arr.csv",arr,delimiter=',')
ix = random.randint(0,19)
iy = random.randint(0,19)
en = one_energy(arr, ix, iy, nmax)
order = get_order(arr, nmax)

f = open("data.txt","w")
f.write(f"ix,iy,en,order\n{ix},{iy},{en},{order}")
f.close()