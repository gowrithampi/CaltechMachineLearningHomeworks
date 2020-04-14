# -*- coding: utf-8 -*-
#Caltech Machine Learning Assignment 2 Question 5 and 6

"""
Created on Mon Mar 23 19:17:21 2020

@author: gthampi
"""
#TakeN= 100.  Use Linear Regression to findgand evaluateEin, the fraction ofin-sample points which got classified incorrectly.
#Repeat the experiment 1000times and take the average (keep theg’s as they will be used again in Problem6). 
#Which of the following values is closest to the averageEin?
#(Closestis theoption that makes the expression|your answer−given option|closest to 0.
# Usethis definition ofclosesthere and throughout.)

import numpy as np
import pandas as pd
from numpy import random
#from scipy.stats import binom
from datetime import datetime
start=datetime.now()


#Create X0, X1, X2 - dataframe
nx = 1000
#size of out of sample dataset
nx_out = 1000
nlines = 1000

Xvec = pd.DataFrame({'x0':np.array([1]*nx), 'x1':random.uniform(-1,1,nx) , 'x2':random.uniform(-1,1,nx) } )


Xvec_out = pd.DataFrame({'x0':np.array([1]*nx_out), 'x1':random.uniform(-1,1,nx_out) , 'x2':random.uniform(-1,1,nx_out) } )


linvec  = pd.DataFrame({'x1lin1' : random.uniform(-1,1,nlines)  , 'x2lin1' : random.uniform(-1,1,nlines) ,
                                   'x1lin2' : random.uniform(-1,1, nlines)  , 'x2lin2' : random.uniform(-1,1,nlines)  })

Xvalues = Xvec.values
Xvalues_out = Xvec_out.values
Xpseudo = np.linalg.inv(Xvalues.transpose().dot(Xvalues)).dot(Xvalues.transpose())
#loop through the line vec create a dataset, run regression 

Ervec = []
Ervec_out = []
for linenumber, line in linvec.iterrows(): 
    
    #these are deterministic y values for in and out of sample 
    yvec = np.sign(Xvec['x2']-line['x2lin1'] + (line['x2lin2'] - line['x2lin1'])*(Xvec['x1']-line['x1lin1'])/(line['x1lin2'] - line['x1lin1']))
    
    yvec_out = np.sign(Xvec_out['x2']-line['x2lin1'] + (line['x2lin2'] - line['x2lin1'])*(Xvec_out['x1']-line['x1lin1'])/(line['x1lin2'] - line['x1lin1']))
    
    #this is my chosen hypothesis g
    Betahat = Xpseudo.dot(yvec)
    
    yhat = np.sign(Xvalues.dot(Betahat.transpose()))
    
    #generate predicted y values for the out of sample dataset
    yhat_out = np.sign(Xvalues_out.dot(Betahat.transpose()))
    
    Ervec.append(np.mean(np.absolute(yhat-yvec)/2))
    
    Ervec_out.append(np.mean(np.absolute(yhat_out -yvec_out)/2))
    
    
print(np.mean(Ervec))   
print(np.mean(Ervec_out))