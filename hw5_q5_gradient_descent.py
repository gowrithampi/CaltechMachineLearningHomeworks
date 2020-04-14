# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:49:09 2020

@author: gthampi
"""

import numpy as np

#H5 question 567 Caltech Machine Learning

def error(u,v):
    return pow(u*np.exp(v)-2*v*np.exp(-u),2)
    

def partialderivative_ewrtu(u,v):
    return 2*(u*np.exp(v)-2*v*np.exp(-u))*(np.exp(v) + 2*v*np.exp(-u))

def partialderivative_ewrtv(u,v):
    return 2*(u*np.exp(v)-2*v*np.exp(-u))*(u*np.exp(v)-2*np.exp(-u))

def gradientdescent(u,v, eta):
    count = 0
    while error(u,v)>pow(10,-14):
         u2 = u - eta*(partialderivative_ewrtu(u,v))
         v2 = v - eta*(partialderivative_ewrtv(u,v))
         u = u2
         v = v2
         count = count + 1
         print(u,v,error(u,v))
    return(u,v,count)
    
def coordinatedescent(u,v,eta):
    count = 0
    while count<30:
        if(count%2==0):
           u = u - eta*(partialderivative_ewrtu(u,v))
        else:
           v = v - eta*(partialderivative_ewrtv(u,v))
        count = count +1
        print(u,v,error(u,v))
    return(u,v,count)


print(coordinatedescent(1.0,1.0,0.1))
        
    