# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:14:24 2020

@author: gthampi
"""

### caltech assignment 2, Question 8 - 10 non linear transformation 

import pandas as pd
import numpy as np
from numpy import random

#Create Xvec 

nx = 10000

xvec = pd.DataFrame({'x0':[1]*nx , 'x1': random.uniform(-1,1,nx), 'x2' : random.uniform(-1,1,nx)} )

#create non linear version of the XVec dataset
xvecnl = xvec.copy()
xvecnl['x1x2'] = xvec['x1']*xvec['x2']
xvecnl['x1sq'] = xvec['x1']*xvec['x1']
xvecnl['x2sq'] = xvec['x2']*xvec['x2']



xvec_values = xvec.values
xvecnl_values = xvecnl.values
pseudox = np.linalg.inv(xvec_values.transpose().dot(xvec_values)).dot(xvec_values.transpose())
pseudoxnl = np.linalg.inv(xvecnl_values.transpose().dot(xvecnl_values)).dot(xvecnl_values.transpose())


#generate deterministic y values according to given objective function
yvec = np.sign(np.power(xvec['x1'],2) + np.power(xvec['x2'],2) -0.6)

#simulate noise by flipping the sign 10 percent of the time 
yvec_noisy = list(map(lambda y: -y if random.uniform(0,1) < 0.1 else y, yvec))

#chosen 
betahat = pseudox.dot(yvec_noisy)
betahatnl = pseudoxnl.dot(yvec_noisy)

print(betahat, betahatnl)




