# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:35:25 2017

@author: frederique
"""

import math
import numpy as np

import matplotlib.pyplot as plt
from random import randint
import scipy.io
import scipy.signal as ss
import scipy.linalg as sl

def displaySignal(x,color,label):
    """ Display input signal""" 
    
    plt.figure()
    
    plt.plot(x,color, label=label)
    
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('y')
    plt.xlabel("x")

    plt.show()

def displaySignals(x1,x2,x3):
    """ Display input signal""" 
    
    plt.figure()
    
    plt.plot(x1,'r')
    plt.plot(x2,'b')
    plt.plot(x3,'g')
    
    plt.plot(x1,'r', label='x1')
    plt.plot(x2,'b', label='x2')
    plt.plot(x3,'g', label='x3')
    
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('y')
    plt.xlabel("x")

    plt.show()
    

def displayInfo(M):
    """ """
    
    # Input Signal
    x=M['x']
    #print("\nx =",x)
    
    dimX=M['dimx']
    print("\ndimX =",dimX)
    dimY=M['dimy']
    print("\ndimY =",dimY)
    
    # Transfo matrix
    A=M['A']
    #print("\nA =",A)
    
    #Wavelet signal
    a=M['a']
    #print("\na =",a)
    dimA=M['dima']
    print("\ndimA =",dimA)
    
    return x,dimX,A,a,dimA,dimY


def computeConvolution(inputSig, TransfoMat):
    """ """
    #imfilter/convolve2d
    conv=ss.convolve(inputSig, TransfoMat)
    
    return conv

def soustractSignals(x1,x2):
    sous=x1-x2
    return sous


def computeSingularValueDecomposition(mat):
    """ """
    unitaryMatrixU,singularValues,unitaryMatrixU=sl.svd(mat)
    
    return unitaryMatrixU,singularValues,unitaryMatrixU

def computeConditionnement(mat):
    """ Comput conditionnement of matrix (max(singular vlaues)*max(1/singular values)
    <=> max(singular vlaues)/min(singular values) """
    
    unitaryMatrixU,singularValues,unitaryMatrixV=computeSingularValueDecomposition(A)
    #print('\nUnitary Matrix Left = \n',unitaryMatrixU)
    #print('\nSingular Values =\n',singularValues)
    #print('\nUnitary Matrix Right = \n',unitaryMatrixV)
    displaySignal(singularValues,'r','Singular Values')
    
    valueMax=singularValues[0]
    valueMin=singularValues[singularValues.shape[0]-1]
    
    #print('\nvalueMax : ',valueMax)
    #print('\nvalueMin : ',valueMin)
    
    # Conditionnement
    cond=valueMax/valueMin
    
    return cond
    
def addNoise(signal,length, s):
    """ Adds noise to a column signal"""
    
    noise = np.random.randn(length)*s
#    print(noise)
#    print('noise dimensions :', noise.shape)
#    print('signal dimensions :', signal.T.shape)
    
    noisedSignal=noise+signal.T  
#    print('noised signal dimensions :', noisedSignal.shape)
    
    return noise, noisedSignal
    
def computePseudoInverse(y,A):
    """ Computes pseudo inverse (solution to mean squares)"""
    
    pseudo=(np.linalg.inv((A.T).dot(A))).dot(A.T)
#    print('pseudo dimensions : ',pseudo.shape)
#    print('y dimensions : ',y.shape)
    
    res=pseudo.dot(y)

    return res

def computePseudoInverseWithAlpha(y,A, alpha):
    """ Computes pseudo inverse (solution to mean squares) with alpha, regularisation parameter"""
    
#    print('\nshape of A',A.shape)
    pseudo=(np.linalg.inv((A.T).dot(A))+alpha*np.eye((A.shape[0]))).dot(A.T)
#    print('pseudo dimensions : ',pseudo.shape)
#    print('y dimensions : ',y.shape)
    
    res=pseudo.dot(y)

    return res
    
    
#def computeFourier(x):
#    """ """
#    
#    # Compute fourier transformation of matrix x. Beware complex numbers.
#    res=np.fft.fft(x)
#    
#    # Compute module of results.
#    res_mod=abs(res)
#    
#    res_trie=tri(res_mod)
#    
#    
#    return res_trie
#    
#
#def tri(aVect):
#    """ """
#    newVect=[]
#    i=0
#    while(i<aVect.shape[0]):
#        newVect.append(min(aVect))
#        index=aVect.argmin
#        aVect[index]=999
#        i+=1
#        
#    print(newVect)
#    
#    
#    
#    return newVect

if __name__=='__main__':

    ## QUESTION 1.1
    kramer = scipy.io.loadmat('kramer.mat')
    #print("\nmat = \n",kramer)
   
    #display signal(kramer) and ransfo of signal by A
    x,dimX,A,a,dimA,dimY=displayInfo(kramer)
   
    displaySignal(x,'b','x')
    displaySignal(A.dot(x),'r','y=Ax')
    
    #Convolve signal and display
    yc=computeConvolution(x,a)
    #print('\nConvolution : ',myConv)
    
    displaySignal(yc,'g','a*x')
    displaySignals(x,A.dot(x),yc)
    
    # Soustract Ax to convolution and display
    new=soustractSignals(A.dot(x),yc)
    displaySignal(new,'r','Ax-ax')
   
    #==== POURQUOI LE PB EST-IL SUR-CONTRAINT ??
    # A*x donne plus d'obs qu'il y a de parametres dans x
   
   
    ## QUESTION 1.2
    myCond=computeConditionnement(A)
    print('myCond = ',myCond)
    
    
    ## QUESTION 1.3
    interv= [0.000001,0.00001,0.0001,0.001,0.1,1,2,3,4,5,6,7,8,9,10]
    interv2=np.arange(0.1,1,0.1)
    interv3=np.arange(0.1,0.4,0.05)
    
    
    
#    for s in interv3:
#        myNoise, myNoisedSignal=addNoise(A.dot(x),dimY, s)
#        #displaySignal(myNoise,'b','noise')
#        #displaySignals(x,myNoise,myNoisedSignal.T)
#        
#        # Mean Squares
#        unknownParam=computePseudoInverse(myNoisedSignal.T,A)
#        #print('Unknown Parameters =\n', unknownParam)
#        
#        print('\ns : ',s)
#        displaySignals(x,unknownParam,0)
#        # above  0.1, xth et xdet too far
#        # seuil : 0.15
    
    
    
    # conditionnement lié àécart-type bruit
        
    
    ## QUESTION 1.4
    print('\nRICKER\n')
    ricker = scipy.io.loadmat('ricker.mat')
    
    x_ricker,dimX_ricker,A_ricker,a_ricker,dimA_ricker,dimY_ricker=displayInfo(ricker)
    #displaySignal(x_ricker,'b','x_ricker')
    #displaySignal(A_ricker.dot(x_ricker),'r','y=Ax_ricker')
    
    yc_ricker=computeConvolution(x_ricker,a_ricker)
    #displaySignal(yc_ricker,'g','a*x_ricker')
    displaySignals(x_ricker,A_ricker.dot(x_ricker),yc_ricker)
    
    # soustraction impossible because of borders    
    
    myCond_ricker=computeConditionnement(A_ricker)
    print('myCond = ',myCond)

    interv_ricker= [0.000001,0.00001,0.0001,0.001,0.1,1,2,3,4,5,6,7,8,9,10]
    interv2_ricker=np.arange(0.0001,0.001,0.0001)
#    for s in interv2_ricker:
#        myNoise_ricker, myNoisedSignal_ricker=addNoise(A_ricker.dot(x_ricker),dimY_ricker, s)
#        #displaySignal(myNoise,'b','noise')
#        #displaySignals(x,myNoise,myNoisedSignal.T)
#        
#        # Mean Squares
#        unknownParam_ricker=computePseudoInverse(myNoisedSignal_ricker.T,A_ricker)
#        #print('Unknown Parameters =\n', unknownParam)
#        
#        print('\ns : ',s)
#        displaySignals(x_ricker,unknownParam_ricker,0)
#        # seuil : 0.0008
        
    ## QUESTION 1.5
#    myFourier=computeFourier(a)
#    print('\nmyFourier : ',myFourier)
    
    
    
    ## QUESTION 2.1
    myNoise_ricker, myNoisedSignal_ricker=addNoise(A_ricker.dot(x_ricker),dimY_ricker, 0.00001)
    intera=[0.00001,0.0001,0.001,0.01,0.1,1]    
    intera2=np.arange(0.001,0.016,0.001)
    
#    for alpha in intera2:
#        print('\nalpha : ',alpha)
#        unknownParam_ricker2=computePseudoInverseWithAlpha(A_ricker.dot(x_ricker),A_ricker, alpha)
#        displaySignals(x_ricker,unknownParam_ricker2,0)
#     #alpha=0.004
     
    for alpha in intera:
        print('\nalpha : ',alpha)
        unknownParam_ricker2=computePseudoInverseWithAlpha(myNoise_ricker,A_ricker, alpha)
        displaySignals(x_ricker,unknownParam_ricker2,0)
    #alpha=?
        
        
        
        
    ## QUESTION 2.2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    