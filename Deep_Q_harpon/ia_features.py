



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:46:06 2019

@author: odwin
"""

import gym
import time
import numpy as np
import random as rd
import Perceptron_Q_Learning as per 
import Harpon_deep_Q as q
#%%

render = False
env = gym.make('Pong-v0')
env.reset()
# if render: env.render()

reward_global = 0.0

reward_survie = 0.0 #DLC Survie

reward_total = 0.0 


def init_game():
#Initialisation du jeu : les 20 premières frames ne servent à rien
    for _ in range(21):
        # if render: env.render()
        observation, reward, done, info = env.step(0) # take no action (2 is up, 5 is down)
    #     
    # env.reset()
    # env.close()
    pre = observ_process(observation)
    state0 = observ_process(observation,pre)
    return state0

def observ_process(observation, obs = [0,0,0,0,0,0,0,0]): #crops the image and selects the red channel
    observationR=[[0 for k in range(len(observation[0]))] for l in range(34,len(observation)-15)]
    for k in range(34,len(observation)-15):    
        for l in range(len(observation[0])):
            observationR[k-34][l]=observation[k][l][0]
    c = observationR
    d = np.array(c).T
    n = len(d)
    m = len(d[0])
    dd = np.array([[0 for j in range(m-1)] for i in range(n)])
    for i in range(n):
        dd[i] = d[i][0:m-1]
    m = len(dd[0])
    n = len(dd)
    d18 = np.array([dd[18][m-i-1] for i in range(m)])
    e4 = m - dd[18].argmax()
    e3 = d18.argmax()
    d140 = np.array([dd[140][m-i-1] for i in range(m)])
    e2 = m - dd[140].argmin()
    e1 = d140.argmin()
    uf = True
    e5 = 0
    e6 = 0
    for i in range(20,140):
        if dd[i].max() != 144 and uf :
            e6 = m - dd[i].argmax()
            e5 = 160 - i
            uf = False
    e7 = obs[4]
    e8 = obs[5]
    e1 = e1/m-0.5
    e2 = e2/m-0.5
    e3 = e3/m-0.5
    e4 = e4/m-0.5
    e5 = e5/m-0.5
    e6 = e6/m-0.5
    return [e1,e2,e3,e4,e5,e6,e7,e8]

def traite(etat,done,reward):
    global reward_global
    if done :
        a  = env.reset()
    if etat[4] == -0.5 and etat[5] == -0.5 :
        toobs, reward, done, info = env.step(2)
        #env.render()
        if done :
            a  = env.reset()
        if reward != 0 :
            reward_global=reward
        observation, reward, done, info = env.step(5)
        #env.render()
        if done :
            a  = env.reset()
        if reward != 0 :
            reward_global=reward
        obs = observ_process(toobs,toobs)
        etat = observ_process(observation,obs)
        return traite(etat,done,reward)
    elif reward != 0 :
        toobs, reward, done, info = env.step(2)
        #env.render()
        if done :
            a  = env.reset()
        observation, reward, done, info = env.step(5)
        #env.render()
        if done :
            a  = env.reset()
        obs = observ_process(toobs,toobs)  
        etat = observ_process(observation,obs)
        return traite(etat,done,reward)
    else:
        return etat
    

def A_up(R,Q,A,obs):
    observation, reward, done, info = env.step(2)
    #env.render()
    global reward_global
    global reward_survie #DLC Survie
    global reward_total #DLC Survie
    reward_survie = reward_survie + 1 #DLC Survie
    reward_global=reward
    etat = observ_process(observation,obs)
    res = traite(etat,done,reward)
    if reward_global < 0: #DLC Survie
        reward_survie = 0  #DLC Survie
        print("reset survie")
    reward_total = reward_global+ 0.8/(1+np.exp((250-reward_survie)/50))  #DLC Survie
    return  res
    
def A_down(R,Q,A,obs):
    observation, reward, done, info = env.step(5)
    #env.render()
    global reward_global
    global reward_survie #DLC Survie
    global reward_total #DLC Survie
    reward_survie = reward_survie + 1 #DLC Survie
    reward_global=reward
    etat = observ_process(observation,obs)
    res = traite(etat,done,reward) 
    if reward_global < 0: #DLC Survie
        reward_survie = 0  #DLC Survie
        print("reset survie")
    reward_total = reward_global+ 0.8/(1+np.exp((250-reward_survie)/50))  #DLC Survie
    return  res

def R(p):
    #return reward_global #à décomenter
    return reward_total #DLC survie 

def modify(x):
    return (100000*x+0.1)/100001

def deep_pong(state0):
    A = [A_up,A_down]
    s0 = state0
    memoire = 1000
    it = 50000
    neural_it = 1
    reseau = [32,32]
    QW,QB = q.deepQlearning(A,s0,R,chooseDeepPong,memoire,it,neural_it,reseau,Tlim = 10e9,phi = q.phibase,gamma = 0.6,rate = 0.0001,opt = 0.3,modify = modify,QW = W,QB = B)
    return QW,QB

def chooseDeepPong(p,R,Q,reseau,A,opt):
    (QW,QB) = Q
    r = rd.random()
    if r < opt :
        rr = rd.randint(0,len(A)-1)
        return A[rr]
    else:
        fp = per.front_prop(p,reseau,QW,QB,per.tanh)[-1]
        res = fp.argmax()
        return A[res]

def test(W, B):
    print("a")
    A = [A_up,A_down]
    reseau = [32,32,2]
    print("b")
    for i in range(50):
        ss = q.frontprop_deep(A,state0,R,(W,B),reseau,chooseDeepPong,0)[0][-1]
            


B,W = [],[]
state0 = init_game()
print(state0)
print(len(state0))
W,B = deep_pong(state0)
W = np.load('Wsimple.npy')
B = np.load('Bsimple.npy')
np.save("Wsimple",W)
np.save("Bsimple",B)
#test(W,B)



    
# for _ in range(1000):
#     if rd.randint(1,2)==1:
#         A_up(observation, reward, done, info)
#         print("up")
#     else:
#         A_down(observation, reward, done, info)
#         print("down")

env.reset()
env.close()

