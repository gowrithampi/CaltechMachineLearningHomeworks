# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:33:13 2020

@author: gthampi
"""


import pandas as pd
import numpy as np



din = pd.read_table('din.txt', sep = '\s+' , names = ['x1' , 'x2', 'y'])
din['x0']=1
din = din[['x0', 'x1','x2', 'y']]
dout = pd.read_table('dout.txt', sep = '\s+' , names = ['x1' , 'x2', 'y'])
dout['x0'] = 1
dout = dout[['x0', 'x1','x2', 'y']]


#perform non linear transformation
din['x1sq'] = pow(din['x1'],2)
din['x2sq'] = pow(din['x2'],2)
din['x1x2'] = din['x1']*din['x2']
din['absdiff'] = abs(din['x1']-din['x2'])
din['abssum'] = abs(din['x1']+din['x2'])
din = din[['x0', 'x1','x2', 'x1sq','x2sq','x1x2','absdiff','abssum', 'y']]
xin = din.iloc[:,0:7].values
yin = din.iloc[:,8].values
pseudox = np.linalg.inv(xin.transpose().dot(xin)).dot(xin.transpose())
betahat = pseudox.dot(yin)
yhat = np.sign(xin.dot(betahat.transpose()))
ein = sum(abs(yhat -yin)/2)/len(yhat)

#perform non linear transformation on dout
dout['x1sq'] = pow(dout['x1'],2)
dout['x2sq'] = pow(dout['x2'],2)
dout['x1x2'] = dout['x1']*dout['x2']
dout['absdiff'] = abs(dout['x1']-dout['x2'])
dout['abssum'] = abs(din['x1']+dout['x2'])
dout = dout[['x0', 'x1','x2', 'x1sq','x2sq','x1x2','absdiff','abssum', 'y']]
xout = dout.iloc[:,0:7].values
yout = dout.iloc[:,8].values
yout_hat = np.sign(xout.dot(betahat.transpose()))
eout = sum(abs(yout_hat -yout)/2)/len(yout_hat)
print('In and out of sample error without regularization are' ,ein,'and ', eout)

#now with regularization 

def lambdagen(k):
    return pow(10,k)

def betahat_reg(lambda_reg):
    idmat = np.identity(7)
    pseudox_reg = np.linalg.inv(xin.transpose().dot(xin)+lambda_reg*idmat).dot(xin.transpose())
    betahat_reg = pseudox_reg.dot(yin)
    return betahat_reg

def ein_eout(betahat):
    yhat = np.sign(xin.dot(betahat.transpose()))
    ein = sum(abs(yhat -yin)/2)/len(yhat)
    yout_hat = np.sign(xout.dot(betahat.transpose()))
    eout = sum(abs(yout_hat -yout)/2)/len(yout_hat)
    return(ein,eout)

for i in range(-10,10):
    lambda_reg = lambdagen(i)
    betahat= betahat_reg(lambda_reg)
    ein, eout = ein_eout(betahat)
    print('The insample and out of sample error for lambda = 10 power' , i, 'are', ein , eout)
    
    
