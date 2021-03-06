# Library for Psychophysic Bayesian Model fitting
#
# List of functions:
#   Psi(x, mu, s)
#   psyFun(x, al, be, mu, s)
#   llkhd(x, y, n, al, be, mu, s)
#   build_objfun0(PsychFunc, nBlock)
#   build_objfun(PsychFunc1, PsychFunc2, nBlock)
#   
#
#   Copyright (c) 2015 Jonathan Vacher

import numpy as np
import scipy as sp
import scipy.optimize as opt

import time
import sys
from ipyparallel import Client
from IPython.display import display, clear_output

# compute the kernel of a matrix
def null(A, eps=1e-12):
    u, s, vh = sp.linalg.svd(A)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    null_space = sp.compress(null_mask, vh, axis=0)
    return null_space    

# erf function between 0 and 1
def Psi(x,mu,s):
    return (1.0+sp.special.erf((x-mu)/(np.sqrt(2)*s)))/2

# psychometric curve model
def psyFun(x, al, be, mu, s):
    return al + (1.0-al-be)*Psi(x, mu, s)

# log likelihood of the kullback leibler distance between binomial parameters
# parametrized by psyFun
def llkhd(x, y, n, al, be, mu, s):
    pf = psyFun(x, al, be, mu, s)
    return -np.sum( y*np.log(pf)+(n-y)*np.log(1.0-pf) )
    #return -np.sum(np.log(pf**y * (1.0-pf)**(n-y) * sp.special.binom(n,y)))

# prior on marginal errors
def prior_al(al, x):
    if np.all( (0<al) & (al<x) ):
        return 1
    else:
        return 0
# prior on standard dev    
def prior_sig(si):
    if np.all( (0.0005<si) ):
        return 1
    else:
        return 0  
    
# solve the linear conditions between likelihood widths (sig^2) and prior slopes
# two possible constraints minimum norm or minimum norm of the derivative 
def initCond1(xx0, freq=0):
    
    szstar0 = xx0[3,2]**2 / 2.0
    sz0 = xx0[3,:]**2  - szstar0
    
    mu = np.array([ xx0[2,0], xx0[2,1], xx0[2,3], xx0[2,4] ])
    A = np.array([[-sz0[0],0,sz0[2],0,0],[0,-sz0[1],sz0[2],0,0],\
                  [0,0,sz0[2],-sz0[3],0],[0,0,sz0[2],0,-sz0[4]]])
    az00 = np.matrix(sp.linalg.lstsq(A,mu)[0])
    
    if freq.all()==0:
        one = np.matrix(np.ones(5))
        e = np.matrix(1.0/sz0)
        lbd = az00*( (e*one.T)*one ).T / ( e*(5.0*e - (e*one.T)*one ).T  )
        az00 = az00 + lbd*e
    
    else:
        one = np.matrix(np.ones(4))
        e = np.matrix(np.diff(1.0/sz0)/np.diff(freq))
        lbd = (np.diff(az00)/np.diff(freq))*( (e*one.T)*one - 4.0*e ).T / ( e*(4.0*e - (e*one.T)*one ).T  )
        az00 = az00 + lbd*np.matrix(1.0/sz0)
    
    return az00, sz0   

# for overlapping spatial frequencies
# solve the linear conditions between likelihood widths (sig^2) and prior slopes
# two possible constraints minimum norm or minimum norm of the derivative    
def initCond(xx0, xx1, freq=0):
    
    szstar0 = xx0[3,2]**2 / 2
    sz0 = xx0[3,0:5]**2 - szstar0
    szstar1 = xx1[3,2]**2 / 2
    sz1 = xx1[3,0:5]**2 - szstar1
    
    mu = np.array([ xx0[2,0], xx0[2,1], xx0[2,3], xx0[2,4], xx1[2,1], xx1[2,3], xx1[2,4] ])
    A = np.array([[-sz0[0],0,(sz0[2]+sz1[0])/2,0,0,0,0],[0,-sz0[1],(sz0[2]+sz1[0])/2,0,0,0,0],\
                  [0,0,(sz0[2]+sz1[0])/2,-(sz0[3]+sz1[1])/2,0,0,0],[0,0,(sz0[2]+sz1[0])/2,0,-(sz0[4]+sz1[2])/2,0,0],\
                  [0,0,0,-(sz1[1]+sz0[3])/2,(sz1[2]+sz0[4])/2,0,0],[0,0,0,0,(sz1[2]+sz0[4])/2,-sz1[3],0],\
                  [0,0,0,0,(sz1[2]+sz0[4])/2,0,-sz1[4]]])
    az00 = np.matrix(sp.linalg.lstsq(A,mu)[0])
    
    sz00 = np.zeros(7)
    sz00[0:2] = sz0[0:2]
    sz00[2:5] =  (sz1[0:3] + sz0[2:5])/2.0
    sz00[5:7] = sz1[3:5]
    
    if freq.all()==0:
        one = np.matrix(np.ones(7))
        e = np.matrix(1/sz00)
        lbd = az00*( (e*one.T)*one ).T / ( e*(7*e - (e*one.T)*one ).T  )
        az00 = az00 + lbd*e
    
    else:
        one = np.matrix(np.ones(6))
        e = np.matrix(np.diff(1/sz00)/np.diff(freq))
        lbd = (np.diff(az00)/np.diff(freq))*( (e*one.T)*one - 6*e ).T / ( e*(6*e - (e*one.T)*one ).T  )
        az00 = az00 + lbd*np.matrix(1/sz00)
    
    return az00, sz00

# solve the linear conditions between likelihood widths (sig^2) and constant prior slopes
# two possible constraints minimum norm or minimum norm of the derivative 
def initCond1Aconst(xx0, freq=0):
    
    szstar0 = xx0[3,2]**2 / 2
    sz00 = xx0[3,0:5]**2 - szstar0
   
    az00 = np.mean((sz00[2] - sz00[0:5])/xx0[2,:])
    
    return az00, sz00  

# for overlapping spatial frequencies
# solve the linear conditions between likelihood widths (sig^2) and constant prior slopes
def initCondAconst(xx0, xx1, freq=0):
    
    szstar0 = xx0[3,2]**2 / 2
    sz0 = xx0[3,0:5]**2 - szstar0
    szstar1 = xx1[3,2]**2 / 2
    sz1 = xx1[3,0:5]**2 - szstar1
    
    sz00 = np.zeros(7)
    sz00[0:2] = sz0[0:2]
    sz00[2:5] =  (sz1[0:3] + sz0[2:5])/2.0
    sz00[5:7] = sz1[3:5]
    az00 = np.mean(0.5*(sz00[2] - sz00[0:5])/xx0[2,:] + 0.5*(sz00[4] - sz00[2:7])/xx1[2,:])
    
    
    return az00, sz00

# solve the linear conditions between likelihood widths (sig^2) and affine prior slopes
# two possible constraints minimum norm or minimum norm of the derivative 
def initCond1Alin(xx0, freq=0):
    
    szstar0 = xx0[3,2]**2 / 2
    sz00 = xx0[3,0:5]**2 - szstar0
       
    mu = np.array([ xx0[2,0], xx0[2,1], xx0[2,3], xx0[2,4] ])
    A = np.array([[sz00[2]*freq[4] - sz00[0]*freq[2], sz00[2] - sz00[0]],\
                  [sz00[2]*freq[4] - sz00[1]*freq[3], sz00[2] - sz00[1]],\
                  [sz00[2]*freq[4] - sz00[3]*freq[5], sz00[2] - sz00[3]],\
                  [sz00[2]*freq[4] - sz00[4]*freq[6], sz00[2] - sz00[4]]])
    az00 = sp.linalg.lstsq(A,mu)[0].squeeze()
    
    return az00, sz00  

# for overlapping spatial frequencies
# solve the linear conditions between likelihood widths (sig^2) and affine prior slopes
def initCondAlin(xx0, xx1, freq=0):
    
    szstar0 = xx0[3,2]**2 / 2
    sz0 = xx0[3,0:5]**2 - szstar0
    szstar1 = xx1[3,2]**2 / 2
    sz1 = xx1[3,0:5]**2 - szstar1
    
    sz00 = np.zeros(7)
    sz00[0:2] = sz0[0:2]
    sz00[2:5] =  (sz1[0:3] + sz0[2:5])/2.0
    sz00[5:7] = sz1[3:5]
    
    mu = np.array([ xx0[2,0], xx0[2,1], xx0[2,3], xx0[2,4], xx1[2,1], xx1[2,3], xx1[2,4] ])
    A = np.array([[sz00[2]*freq[2] - sz00[0]*freq[0], sz00[2] - sz00[0]],\
                  [sz00[2]*freq[2] - sz00[1]*freq[1], sz00[2] - sz00[1]],\
                  [sz00[2]*freq[2] - sz00[3]*freq[3], sz00[2] - sz00[3]],\
                  [sz00[2]*freq[2] - sz00[4]*freq[4], sz00[2] - sz00[4]],\
                  [sz00[4]*freq[4] - sz00[3]*freq[3], sz00[4] - sz00[3]],\
                  [sz00[4]*freq[4] - sz00[5]*freq[5], sz00[4] - sz00[5]],\
                  [sz00[4]*freq[4] - sz00[6]*freq[6], sz00[4] - sz00[6]]])
    
    az00 = sp.linalg.lstsq(A,mu)[0].squeeze()
    
    
    return az00, sz00


# objective function parametrize by likelihood width (sig^2) and prior slope
def build_objfun1(PsychFunc1, speed, nBlock):
    def objfun(param):
        s = speed[:,np.newaxis,np.newaxis] 
        a1 = param[10] 
        b1 = param[11]
        m1 = speed[2]+param[2]*param[7]-param[0:5]*param[5:10]     
        s1 = np.sqrt(param[5:10] + param[7])      
        m1 = m1[np.newaxis, :, np.newaxis]
        s1 = s1[np.newaxis, :, np.newaxis]
        n = 10 
        return llkhd(s, PsychFunc1, n, a1, b1, m1, s1) - np.log( prior_al(param[10:12], 0.05) )\
                        - np.log(prior_sig(param[5:10]))
    return objfun
# objective function parametrize by likelihood width (sig^2) and constant prior slope
def build_objfun1_Aconst(PsychFunc1, speed, nBlock):
    def objfun(param):
        s = speed[:,np.newaxis,np.newaxis] 
        a1 = param[6] 
        b1 = param[7]
        m1 = speed[2]+param[0]*param[3]-param[0]*param[1:6]     
        s1 = np.sqrt(param[1:6] + param[3])      
        m1 = m1[np.newaxis, :, np.newaxis]
        s1 = s1[np.newaxis, :, np.newaxis]
        n = 10 
        return llkhd(s, PsychFunc1, n, a1, b1, m1, s1) - np.log( prior_al(param[6:8], 0.05) )\
                        - np.log(prior_sig(param[1:6]))
    return objfun

# objective function parametrize by likelihood width (sig^2) and linear prior slope
def build_objfun1_Alin(PsychFunc1, speed, freq, nBlock):
    def objfun(param):
        s = speed[:,np.newaxis,np.newaxis] 
        a1 = param[7] 
        b1 = param[8]
        m1 = speed[2]+(param[0]*freq[4]+param[1])*param[4]-(param[0]*freq[2:]+param[1])*param[2:7]     
        s1 = np.sqrt(param[2:7] + param[4])      
        m1 = m1[np.newaxis, :, np.newaxis]
        s1 = s1[np.newaxis, :, np.newaxis]
        n = 10 
        return llkhd(s, PsychFunc1, n, a1, b1, m1, s1) - np.log( prior_al(param[7:9], 0.05) )\
                        - np.log(prior_sig(param[2:7])) #+ param[1]**2
    return objfun


# for overlapping spatial frequencies
# objective function parametrize by likelihood width (sig^2) and prior slope
def build_objfun(PsychFunc1, PsychFunc2, speed, nBlock):
    def objfun(param):
        s = speed[:,np.newaxis,np.newaxis] 
        a1 = param[14] 
        b1 = param[15]
        a2 = param[16] 
        b2 = param[17] 
        m1 = speed[2]+param[2]*param[9]-param[0:5]*param[7:12]     
        s1 = np.sqrt(param[7:12] + param[9])      
        m2 = speed[2]+param[4]*param[11]-param[2:7]*param[9:14]     
        s2 = np.sqrt(param[9:14] + param[11])
        m1 = m1[np.newaxis, :, np.newaxis]
        m2 = m2[np.newaxis, :, np.newaxis]
        s1 = s1[np.newaxis, :, np.newaxis]
        s2 = s2[np.newaxis, :, np.newaxis]
        n = 10 
        return llkhd(s, PsychFunc1, n, a1, b1, m1, s1) + llkhd(s, PsychFunc2, n, a2, b2, m2, s2) \
                - np.log( prior_al(param[14:18], 0.05) ) - np.log(prior_sig(param[7:14])) #+ 0.01*np.log(param[0:7].std())
    return objfun

# for overlapping spatial frequencies
# objective function parametrize by likelihood width (sig^2) and constant prior slope
def build_objfun_Aconst(PsychFunc1, PsychFunc2, speed, nBlock):
    def objfun(param):
        s = speed[:,np.newaxis,np.newaxis] 
        a1 = param[8] 
        b1 = param[9]
        a2 = param[10] 
        b2 = param[11] 
        m1 = speed[2]+param[0]*param[3]-param[0]*param[1:6]     
        s1 = np.sqrt(param[1:6] + param[3])      
        m2 = speed[2]+param[0]*param[5]-param[0]*param[3:8]     
        s2 = np.sqrt(param[3:8] + param[5])
        m1 = m1[np.newaxis, :, np.newaxis]
        m2 = m2[np.newaxis, :, np.newaxis]
        s1 = s1[np.newaxis, :, np.newaxis]
        s2 = s2[np.newaxis, :, np.newaxis]
        n = 10 
        return llkhd(s, PsychFunc1, n, a1, b1, m1, s1) + llkhd(s, PsychFunc2, n, a2, b2, m2, s2) \
                - np.log( prior_al(param[8:12], 0.05) ) - np.log(prior_sig(param[1:8])) #+ 0.01*np.log(param[0:7].std())
    return objfun

# for overlapping spatial frequencies
# objective function parametrize by likelihood width (sig^2) and linear prior slope
def build_objfun_Alin(PsychFunc1, PsychFunc2, speed, freq, nBlock):
    def objfun(param):
        s = speed[:,np.newaxis,np.newaxis] 
        a1 = param[9] 
        b1 = param[10]
        a2 = param[11] 
        b2 = param[12] 
        m1 = speed[2]+(param[0]*freq[2]+param[1])*param[4]-(param[0]*freq[:5]+param[1])*param[2:7]     
        s1 = np.sqrt(param[2:7] + param[4])      
        m2 = speed[2]+(param[0]*freq[4]+param[1])*param[6]-(param[0]*freq[2:]+param[1])*param[4:9]     
        s2 = np.sqrt(param[4:9] + param[6])
        m1 = m1[np.newaxis, :, np.newaxis]
        m2 = m2[np.newaxis, :, np.newaxis]
        s1 = s1[np.newaxis, :, np.newaxis]
        s2 = s2[np.newaxis, :, np.newaxis]
        n = 10 
        return llkhd(s, PsychFunc1, n, a1, b1, m1, s1) + llkhd(s, PsychFunc2, n, a2, b2, m2, s2) \
                - np.log( prior_al(param[9:13], 0.05) ) - np.log(prior_sig(param[2:9])) #+ 0.01*np.log(param[0:7].std())
    return objfun

# objective function parametrized by threshold and differential PSE
def build_objfun0(PsychFunc, speed, nBlock):
    def objfun(param):
        s = speed[:, np.newaxis] 
        n = 10
        return llkhd(s, PsychFunc, n, param[0], param[1], param[2]+speed[2], param[3]) \
                -np.log(prior_al(param[0:2], 0.05))
    return objfun


