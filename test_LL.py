'''
implement some tests on the LL functions - check that we see the same behaviour in main branch and altered branches.
'''
from LebwohlLasher import one_energy, get_order
import numpy as np
import pytest
import pandas as pd

# (function) def one_energy(arr: Any, ix: Any, iy: Any, nmax: Any)
#(function) def get_order(arr: Any, nmax: Any) 

# read in created arr 
arr = np.loadtxt('arr.csv', delimiter=',')
data = pd.read_csv("data.txt")
arr = np.reshape(arr, (20,20))
nmax = np.shape(arr)

nmax = nmax[0]
# these indices are randomly generated, as is the array. ie. if this works, then all of it should work
ix = data["ix"].iloc[0]
iy = data["iy"].iloc[0]
en = data["en"].iloc[0]
order = data["order"].iloc[0]

# tests for functions - ensure approx equal due to floating point values
def test_one_energy():
    out = one_energy(arr,ix,iy,nmax)
    assert out == pytest.approx(en, rel=None, abs=None, nan_ok=False)

def test_get_order():
    out = get_order(arr,nmax)
    assert out == pytest.approx(order, rel=None, abs=None, nan_ok=False)
    