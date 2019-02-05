#!/usr/bin/python3
# -*- coding: utf-8 _*_
"""
Created on Mon Fev 4 17:24:00 2019

@author: kagamino
"""

import gym
import time
import numpy as np
import random
import Perceptron_Q_Learning as per 
import Harpon_Q_Learning as q
import matplotlib.pyplot as plt

from conf import render, iterations, exploration_rate
global exploration_rate

# Initialisation
env = gym.make('Pong-v0')
env.reset()
if render: env.render()
actions_values = [0, 2, 5]
reseau = [256, 32, 32, 3]
global W,B
W,B=per.random_w_b([0 for _ in range(256)],reseau)
gamma=0.9

def init_game(env):
#Initialisation du jeu : les 20 premières frames ne servent à rien
    for _ in range(20):
        env.step(0)
    return True

def to_state(observation):
# crops the image and selects the red channel
    observationR = [[0 for k in range(len(observation[0]))] for l in range(35,len(observation)-15)]
    for k in range(35,len(observation)-15):    
        for l in range(len(observation[1])):
            observationR[k-35][l] = observation[k][l][0]
    return observationR
    #.reshape(1,len(observationR)*len(observationR[0]))[0]
    
def resize(observation):
    cropped = to_state(observation)
    downscaled = [[0]*16 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            temp = 0
            for l in range(10):
                for n in range(10):
                    temp += cropped[10*i+l][10*j+n]-144
            downscaled[i][j] = temp/256
    downscaled_flat = [y for x in downscaled for y in x]
    return downscaled_flat

def point(env, observation):
    global exploration_rate
    reward = 0
    states_list=[]
    actions_list=[]
    observation, reward, done, info = env.step(env.action_space.sample())
    while reward == 0:
        # TODO: Compute action by Network
        actions = list(per.front_prop(resize(observation),reseau,W,B,per.tanh)[-1])
        if random.random() < exploration_rate:
            action = random.choice(actions_values)
            observation, reward, done, info = env.step(action)
        else:
            action_temp=actions.index(max(actions))
            action = actions_values[action_temp]
            observation, reward, done, info = env.step(action)
        states_list+=[resize(observation)]
        actions_list+=[action]
        if render:
            env.render()
    return states_list, actions_list, reward, done


def game(env, played):
    global W,B
    env.reset()
    init_game(env)
    observation, reward, done, info = env.step(0)
    scores = [0, 0] # (ia, opponent)
    done = False
    while not done:
        states_list, actions_list, reward, done = point(env, observation)
        played += 1
        if reward == 1:
            n=len(states_list)
            for k in range(n):
                inputs = states_list[n-k]
                actions = per.front_prop(inputs,reseau,W,B,per.tanh)[-1]
                temp = actions_list[n-k]
                th_outputs = actions
                th_outputs[actions_values.index(temp)]=gamma**k
                W,B,_ = per.backprop(inputs,th_outputs,reseau,W,B,per.tanh,per.dtanh)
                
            scores[0] += 1
            
        else:
            
            inputs = states_list[-2]
            actions = per.front_prop(inputs,reseau,W,B,per.tanh)[-1]
            temp = actions_list[-2]
            th_outputs = actions
            th_outputs[actions_values.index(temp)]=-1
            W,B,_ = per.backprop(inputs,th_outputs,reseau,W,B,per.tanh,per.dtanh)
            
            scores[1] += 1
        print("#{:5d}# IA - opp: {:2d} vs {:2d}".format(played, scores[0], scores[1]))
    print("End game")
    return scores, played



# TEST
# def frontprop(observation):
#     return [0.1, 0.2, 0.4]

init_game(env)
played = 0
while played < iterations:
    scores, played = game(env, played)
    exploration_rate = max(0.1,exploration_rate*exploration_rate)