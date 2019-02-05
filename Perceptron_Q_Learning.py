# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 17:55:45 2018

@author: Loic
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd

#%% Neural Network Base

def sigmoid(x):
    return  1/(1+np.exp(-x)) #On choisit cette sigmoide car la dérivée est simple

def dsigmoid(m): #Work in progress do not use
    return m*(np.ones(len(m)) - m)

def tanh(x):
    return 1.7159*np.tanh(2*x/3)

def dtanh(m):
    a = 1.7159
    b = 1/(a**2)
    return a*(2/3)*(np.ones(len(m))-b*m*m)

def front_prop(inputs,reseau,weights,bias,activation = sigmoid):
    #weights is a list of arrays (lines:layers[k],colons:layers[k+1]), bias is a list of arrays 
    #returns the list of results of perceptrons
    #reseau n inclut pas le layer d'entree
    #les weights et biais sont comptés avec leurs colonnes de sortie
    #list_out inclue l'entrée donc len(list_out)=len(reseau)+1
    list_out=[np.zeros(reseau[k]) for k in range(len(reseau))]
    list_out=[np.array(inputs)]+list_out
    for col in range(len(reseau)):
        sig_in=np.dot(list_out[col],weights[col])+bias[col]
        for per in range(len(sig_in)):
            list_out[col+1][per]=activation(sig_in[per])
    return list_out


def backprop(inputs,th_outputs,reseau,weights,bias,activation = sigmoid,derivee = dsigmoid):
    #backpropagation:dc/daijk=dc/dbjk *Yik-1 et dc/bik=SUMj[dc/dbjk+1*aijk+1] *Yik+1(1-Yik+1)
    #matriciellement DAk=DBk.Yk-1 et DBk=Ak+1 . DBk+1*Yk+1 *(1-Yk+1)
    list_out=front_prop(inputs,reseau,weights,bias,activation)    
    grad_weight=[weights[k]*0 for k in range(len(weights))]
    grad_bias=[bias[k]*0 for k in range(len(bias))]
    #init
    grad_bias[-1]= derivee(list_out[-1])*(list_out[-1]-np.array(th_outputs))
    grad_weight[-1]=np.dot(list_out[-2][None].T,grad_bias[-1][None])
    #recurrence
    for col in range(len(list_out)-3,-1,-1):
        grad_bias[col]=np.dot(weights[col+1],grad_bias[col+1])*derivee(list_out[col+1])
        grad_weight[col]=np.dot(list_out[col][None].T,grad_bias[col][None])
    return grad_weight,grad_bias,np.linalg.norm(th_outputs-list_out[-1])/2

def random_w_b(inputs,reseau):
    weights=[2*np.random.random((len(inputs),reseau[0]))-np.ones((len(inputs),reseau[0]))]+[2*np.random.random((reseau[k],reseau[k+1]))-np.ones((reseau[k],reseau[k+1])) for k in range(len(reseau)-1)]
    bias=[np.zeros(reseau[k]) for k in range(len(reseau))]
    return weights, bias

def cost_function(L_inputs,L_th_outputs,reseau,weights,bias,activation=sigmoid):
    cost=0
    for k in range(len(L_inputs)):
        output=front_prop(L_inputs[k],reseau,weights,bias,activation)[-1]
        cost+=np.linalg.norm(L_th_outputs[k]-output)/2
    return cost/len(L_inputs)

#%% Batch learning

def batch_training(L_inputs,L_th_outputs,reseau,weights,bias,rate,iterations,activation = sigmoid,derivee = dsigmoid): 
    error = []
    for k in range(iterations):
        delta_weight = [weights[k]*0 for k in range(len(weights))]
        delta_bias = [bias[k]*0 for k in range(len(bias))]
        cost_tot = 0
        for data in range(len(L_inputs)):
            gw,gb,cost = backprop(L_inputs[data],L_th_outputs[data],reseau,weights,bias,activation,derivee)
            for col in range(len(gw)):
                delta_weight[col] += gw[col]*rate/len(L_inputs)
                delta_bias[col] += gb[col]*rate/len(L_inputs)
            cost_tot += cost/len(L_inputs)
        error.append(cost_tot)
        for col in range(len(weights)):
            weights[col] += -delta_weight[col]
            bias[col] += -delta_bias[col]  
    return weights,bias,error



def minibatch(L_inputs,L_th_outputs,L_inputs_test,L_th_outputs_test,reseau,weights,bias,rate,iterations,batchsize,activation = sigmoid,derivee = dsigmoid):
    #creation de plus petites listes (minibatchs)
    batchs_L_inputs=[]
    batchs_L_th_outputs=[]
    error=[]
    for k in range(len(L_inputs)):
        if k%batchsize==0:
            batchs_L_inputs.append([])
            batchs_L_th_outputs.append([])
        batchs_L_inputs[-1].append(L_inputs[k])
        batchs_L_th_outputs[-1].append(L_th_outputs[k])
    for N in range(iterations):
        for minibatch in range(len(batchs_L_inputs)):
            batch_training(batchs_L_inputs[minibatch],batchs_L_th_outputs[minibatch],reseau,weights,bias,rate,1,activation,derivee)#change weights et bias dans la fonction
        #calcul du coup (oui ca prend longtemps du coup :/ ca double le cout en temps presque faudrait modulariser cout() pour y remedier)
            cost_tot=0
            for data in range(len(L_inputs_test)):
                gw,gb,cost = backprop(L_inputs_test[data],L_th_outputs_test[data],reseau,weights,bias,activation,derivee)
                cost_tot += cost/len(L_inputs_test)
            error.append(cost_tot)
    return weights,bias, error


#%% Stochastic learning

def stochastic_training(total_inputs,total_ouputs,ini_weight,ini_bias,vitesse,reseau,iterations = 1,activation = sigmoid,derivee = dsigmoid):
    #iterations=iterations de l'ensemble du training set
    n = len(total_inputs)
    W = ini_weight
    B = ini_bias
    E = []
    for j in range(iterations):
        for i in range(n):
            I = total_inputs[i]
            O = total_ouputs[i]
            (gW,gB,ee) = backprop(I,O,reseau,W,B,activation,derivee)
            E.append(ee)
            W = [W[i] - vitesse*1959494*gW[i] for i in range(len(W))]
            B = [B[i] - vitesse*1959494*gB[i] for i in range(len(B))]
            #if i/n*100*(j+1)/iterations%1 == 0 :
            #   print(str(i/n*100*(j+1)/iterations) + " % done")
                #print(str(i/n*100*(j+1)/iterations) + " % done")
    return (W,B,E)



def traite_entrees(total_inputs): #It works maggle
    n = len(total_inputs)
    m = len(total_inputs[0])
    moy = np.array([0 for i in range(m)])
    for i in range(n):
        moy = moy + total_inputs[i]
    res = [total_inputs[i] - moy/n for i in range(n)]
    for i in range(n):
        norm = 0
        for j in range(m):
            norm += (res[i][j])**2
        norm = np.sqrt(norm)
        for j in range(m):
            res[i][j] = res[i][j]/norm
    return res

