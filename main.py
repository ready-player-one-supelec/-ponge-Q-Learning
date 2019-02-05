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
import cv2

from conf import render, iterations, exploration_rate

# Initialisation
env = gym.make('Pong-v0')
env.reset()
if render: env.render()
actions_values = [0, 2, 5]
reseau = [100, 32, 32, 3]

def init_game(env):
#Initialisation du jeu : les 20 premières frames ne servent à rien
    for _ in range(1):
        env.step(0)
    return True

def to_state(observation):
# crops the image and selects the red channel
    observationR=[[0 for k in range(len(observation[0]))] for l in range(35,len(observation)-15)]
    for k in range(35,len(observation)-15):    
        for l in range(len(observation[0])):
            observationR[k-35][l]=observation[k][l][0]
    return np.array(observationR).reshape(1,len(observationR)*len(observationR[0]))[0]
    
def resize(observation):
    

def point(env, observation):
    reward = 0
    states=[]
    actions=[]
    while reward == 0:
        # TODO: Compute action by Network
        observation, reward, done, info = env.step(env.action_space.sample())

        actions = frontprop(observation)
        if random.random() < exploration_rate:
            action = random.choice(actions_values)
            observation, reward, done, info = env.step(action)
        else:
            action = actions_values[actions.index(max(actions))]
            observation, reward, done, info = env.step(action)
        states+=[cv2.resize]
        actions+=[action]
        if render:
            env.render()
    return states, actions, reward, done


def game(env, played):
    env.reset()
    observation, reward, done, info = env.step(env.action_space.sample())
    scores = [0, 0] # (ia, opponent)
    done = False
    while not done:
        observation, reward, done = point(env, observation)
        played += 1
        if reward == 1:
            scores[0] += 1
        else:
            scores[1] += 1
        print("#{:5d}# IA - opp: {:2d} vs {:2d}".format(played, scores[0], scores[1]))
    print("End game")
    return scores, played



# TEST
def frontprop(observation):
    return [0.1, 0.2, 0.4]

init_game(env)
played = 0
while played < iterations:
    scores, played = game(env, played)