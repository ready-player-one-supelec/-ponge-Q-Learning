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

env = gym.make('Pong-v0')
env.reset()
# env.render()

# CONST
reward_global = 0.0
it = 2000
# batch_size = 32
neural_it = 1
memoire = 1000000


def init_game():
#Initialisation du jeu : les 20 premières frames ne servent à rien
    for _ in range(21):
        # env.render()
        observation, reward, done, info = env.step(0) # take no action (2 is up, 5 is down)
    #     
    # env.reset()
    # env.close()
    
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
    # env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action (2 is up, 5 is down)
    global reward_global
    reward_global=reward
    if done :
        a  = env.reset()
    while reward != 0 :
        observation, reward, done, info = env.step(2)
        # env.render()
        observation, reward, done, info = env.step(5)
        # env.render()
    return observ_process(observation) 
    
def A_down(observation, reward, done, info):
    observation, reward, done, info = env.step(5)
    # env.render()
    global reward_global
    reward_global=reward
    if done :
        a  = env.reset()
    while reward != 0 :
        observation, reward, done, info = env.step(2)
        # env.render()
        observation, reward, done, info = env.step(5)
        # env.render()
    return observ_process(observation) 

def R(p):
    return reward_global

def deep_pong(state0):
    A = [A_up,A_down]
    s0 = state0
    reseau = [32,16]
    QW,QB = q.deepQlearning(A,s0,R,chooseDeepPong,memoire,it,neural_it,reseau,Tlim = 10e9,phi = q.phibase,gamma = 0.6,rate = 0.0001,opt = 0.3,modify = lambda x: x)
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

state0 = init_game()
print(state0)
print(len(state0))
W,B = deep_pong(state0)
np.save('./save-W', W)
np.save('./save-B', B)

    
# for _ in range(1000):
#     if rd.randint(1,2)==1:
#         A_up(observation, reward, done, info)
#         print("up")
#     else:
#         A_down(observation, reward, done, info)
#         print("down")

env.reset()
env.close()
