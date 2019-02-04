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
import time
import Perceptron_Q_Learning as per 
import Harpon_Q_Learning as q


# CONST
from conf import *

env = gym.make('Pong-v0')
env.reset()
if render: env.render()



def init_game():
#Initialisation du jeu : les 20 premières frames ne servent à rien
    for _ in range(21):
        if render: env.render()
        observation, reward, done, info = env.step(0) # take no action (2 is up, 5 is down)
    #     
    env.reset()
    env.close()
    
    state0 = observ_process(observation)
    return state0

def observ_process(observation): #crops the image and selects the red channel
    observationR=[[0 for k in range(len(observation[0]))] for l in range(34,len(observation)-15)]
    for k in range(34,len(observation)-15):    
        for l in range(len(observation[0])):
            observationR[k-34][l]=observation[k][l][0]
    return np.array(observationR).reshape(1,len(observationR)*len(observationR[0]))[0]

def A_up(observation, reward, done, info):
    observation, reward, done, info = env.step(2)
    if render: env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action (2 is up, 5 is down)
    global reward_global
    reward_global=reward
    if done :
        a  = env.reset()
    while reward != 0 :
        observation, reward, done, info = env.step(2)
        if render: env.render()
        observation, reward, done, info = env.step(5)
        if render: env.render()
    return observ_process(observation) 
    
def A_down(observation, reward, done, info):
    observation, reward, done, info = env.step(5)
    if render: env.render()
    global reward_global
    reward_global=reward
    if done :
        a  = env.reset()
    while reward != 0 :
        observation, reward, done, info = env.step(2)
        if render: env.render()
        observation, reward, done, info = env.step(5)
        if render: env.render()
    return observ_process(observation) 
    
def A_still(observation, reward, done, info):
    observation, reward, done, info = env.step(0)
    if render: env.render()
    global reward_global
    reward_global=reward
    if done :
        a  = env.reset()
    while reward != 0 :
        observation, reward, done, info = env.step(2)
        if render: env.render()
        observation, reward, done, info = env.step(5)
        if render: env.render()
    return observ_process(observation)   

def R(p):
    return reward_global

def modify(x):
    return (100000*x+0.3)/100001

def deep_pong(state0):
    A = [A_up,A_down,A_still]
    s0 = state0
    reseau = [32,32]
    QW,QB = q.deepQlearning(A,s0,R,chooseDeepPong,memoire,it,neural_it,reseau,Tlim = 10e9,phi = q.phibase,gamma = 0.6,rate = 0.0001,opt = 1,modify = modify)
    return QW, QB
    

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

def test(W= [], B =  []):
    if W == [] or B == []:
        W = np.load('./W.npy')
        B = np.load('./B.npy')
    A = [A_up,A_down]
    reseau = [32,32,2]
    for i in range(50):
            ss = q.frontprop_deep(A,state0,R,(W,B),reseau,chooseDeepPong,0)[0][-1]

state0 = init_game()
print(state0)
print(len(state0))
W,B = deep_pong(state0)
np.save('./W', W)
np.save('./B', B)
# test()

            

    
# for _ in range(1000):
#     if rd.randint(1,2)==1:
#         A_up(observation, reward, done, info)
#         print("up")
#     else:
#         A_down(observation, reward, done, info)
#         print("down")

env.reset()
env.close()
