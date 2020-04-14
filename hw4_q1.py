# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:08:26 2020

@author: gthampi
"""

from scipy.optimize import fsolve
import numpy as np

def f(N):
    return 8/N*np.log((16*N**10)/0.05) - 5*10**-4

x = fsolve(f, 40000)           

print("The root x is approximately x=%21.19g" % x)
