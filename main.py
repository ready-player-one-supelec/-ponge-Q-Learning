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

from conf import render, iterations

# Initialisation
env = gym.make('Pong-v0')
env.reset()
if render: env.render()

def init_game(env):
#Initialisation du jeu : les 20 premières frames ne servent à rien
    for _ in range(20):
        env.step(0)
    return True

def to_state(observation):
# crops the image and selects the red channel
    observationR=[[0 for k in range(len(observation[0]))] for l in range(34,len(observation)-15)]
    for k in range(34,len(observation)-15):    
        for l in range(len(observation[0])):
            observationR[k-34][l]=observation[k][l][0]
    return np.array(observationR).reshape(1,len(observationR)*len(observationR[0]))[0]

def point(env):
    reward = 0
    while reward == 0:
        # TODO: Compute action by Network
        observation, reward, done, info = env.step(env.action_space.sample())
        if render:
            time.sleep(0.01)
            env.render()
    return reward, done


def game(env, played):
    scores = [0, 0] # (ia, opponent)
    done = False
    while not done:
        reward, done = point(env)
        played += 1
        if reward == 1:
            scores[0] += 1
        else:
            scores[1] += 1
        print("#{:5d}# IA - opp: {:2d} vs {:2d}".format(played, scores[0], scores[1]))
    env.reset()
    print("End game")
    return scores, played

# TEST
init_game(env)
played = 0
while played < iterations:
    scores, played = game(env, played)