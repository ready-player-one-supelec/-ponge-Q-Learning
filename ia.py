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

env = gym.make('Pong-v0')
env.reset()
for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action (2 is up, 5 is down)
env.reset()
env.close()
