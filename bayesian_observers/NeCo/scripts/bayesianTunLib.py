import numpy as np
import scipy as sp
from scipy import special
from scipy import signal
import matplotlib.pyplot as plt
from cycler import cycler

def find_idx_nearest_val(array, value):
    idx_sorted = np.argsort(array)
    sorted_array = np.array(array[idx_sorted])
    idx = np.searchsorted(sorted_array, value, side="left")
    if idx >= len(array):
        idx_nearest = idx_sorted[len(array)-1]
    elif idx == 0:
        idx_nearest = idx_sorted[0]
    else:
        if abs(value - sorted_array[idx-1]) < abs(value - sorted_array[idx]):
            idx_nearest = idx_sorted[idx-1]
        else:
            idx_nearest = idx_sorted[idx]
    return idx_nearest

def Psi(x,mu,sig):
    return (1.0+sp.special.erf((x-mu)/(np.sqrt(2.0)*sig)))/2.0



























