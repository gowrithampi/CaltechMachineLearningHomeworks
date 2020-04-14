# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:11:52 2020

@author: gthampi
"""

## hw5 question 8 and 9
##implements stochastic gradient descent for logistic regression 
from collections import namedtuple
from numpy import random
import random as rnd
import numpy as np
import pandas as pd
import math
line = namedtuple('line', 'x1 y1 x2 y2')

def generate_line():
    return line(random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1))

 
##pass an equation of a line and number of rows
def generate_din(line,nx):    
   din = pd.DataFrame({'x0':np.array([1]*nx), 'x1':random.uniform(-1,1,nx) , 'x2':random.uniform(-1,1,nx) } )
   yvec = np.sign(din['x2']-line.y1 + (line.y2 - line.y1)*(din['x1']-line.x1)/(line.x2 - line.x1))
   din['y'] = yvec
   return din

def generate_dout(line,nx):
   dout = pd.DataFrame({'x0':np.array([1]*nx), 'x1':random.uniform(-1,1,nx) , 'x2':random.uniform(-1,1,nx) } )
   yvec = np.sign(din['x2']-line.y1 + (line.y2 - line.y1)*(din['x1']-line.x1)/(line.x2 - line.x1))
   dout['y'] = yvec
   return din

def distance_w(w1,w2):
   return pow(pow(w2[2] - w1[2],2) + pow(w2[1] -w1[1],2) + pow(w2[0]-w1[0],2),0.5)

def gradient_error(y,x,w):
    return - y*x/(1+ np.exp(y*w.transpose().dot(x)))

def error(y,x,w):
    s = 0
    for i in range(1,len(x)):
        s = s + math.log(1 + np.exp(-y[i]*w.transpose().dot(x[i])))
        
    error = s/int(len(x))
    return error

    
def stochastic_gradient_descent(din,eta):
    
    w = np.zeros(3)
    wnew = w
    epochs = 0
    #create a random combination to make in stochastic
    ordered = np.arange(0,len(din))
    
    
    #pick the first and start gradient descent till epoch
    
    while True:
        randomized = rnd.sample(list(ordered),len(din))
        w = wnew
        for i in range(0,len(din)):
            
            x = np.array(din.iloc[randomized[i]][:-1])
            y = int(din.iloc[randomized[i]][3])        
            wnew = wnew - eta*gradient_error(y,x,w)
        epochs = epochs + 1  
        if distance_w(w,wnew) < 0.01:
            break
        print(wnew, epochs)
        
    return w, epochs
        
        
        
    

##generate a random line to be used as decision surface
decision_line = generate_line()

## generate a dataset for input using the random line as decision surface
din = generate_din(decision_line, 100)

## generate a dataset for testing the our of sample error using a random line as a decision surface
dout = generate_dout(decision_line, 100)

x=np.array(dout.iloc[:,:-1])
y = np.array(dout.iloc[:,-1])
sgd = stochastic_gradient_descent(din,0.01)
w=sgd[0]
print(error(y,x,w))
print(gradient_error(1,np.array([1,1.1,1.2]),np.array([1,1,1])))