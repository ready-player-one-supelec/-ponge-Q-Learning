#!/usr/bin/python3
# -*- coding: utf-8 _*_
"""
Created on Mon Fev 4 17:24:00 2019

@author: kagamino
"""

import gym
import time
import numpy as np
import tensorflow as tf
import random

from cnn import cnn_model
from conf import render, iterations, exploration_rate, verbose

# Initialisation
env = gym.make('Pong-v0')
env.reset()
if render: env.render()
actions_values = {
    "REST": 0,
    "UP": 2,
    "DOWN": 5
}

def init_game(env):
#Initialisation du jeu : les 20 premières frames ne servent à rien
    for _ in range(20):
        env.step(0)
    return True

def to_state(observation):
# crops the image and selects the red channel
    observationR=[[0 for k in range(len(observation[0]))] for l in range(35,len(observation)-15)]
    for k in range(35,len(observation)-15):    
        for l in range(len(observation[0])):
            observationR[k-35][l]=observation[k][l][0]
    return np.array(observationR)#.reshape(1,len(observationR)*len(observationR[0]))[0]

def point(env, observation):
    reward = 0
    observations = []
    while reward == 0:
        # TODO: Compute action by Network
        observation = to_state(observation)
        observation = tf.constant(observation)
        observations.append(observation)
        # input_fn = tf.estimator.inputs.numpy_input_fn(observation, shuffle=False)
        for smth in classifier.predict(input_fn=lambda: observation):
            print(smths)
        actions = frontprop(observation)

        if random.random() < exploration_rate:
            action = random.choice(list(actions_values.values()))
            observation, reward, done, info = env.step(action)
        else:
            action = actions_values[actions.index(max(actions))]
            observation, reward, done, info = env.step(action)
        if render:
            # time.sleep(0.01)
            env.render()
        # TODO: Compute data to feed network
        if reward == 1:
            actions.append(action)
            print("action gagnante")
        elif reward == -1:
            pass
    return observation, reward, done


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
        if verbose: print("#{:5d}# IA - opp: {:2d} vs {:2d}".format(played, scores[0], scores[1]))
    if verbose: print("End game")
    return scores, played

# TEST
def frontprop(observation):
    return [0.1, 0.2, 0.4]

init_game(env)
played = 0
sess = tf.Session()
classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir="./tmp")
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)
# sess.run(classifier.)
while played < 1:
    scores, played = game(env, played)

tf.train.Saver().save(sess, 'model.ckpt')
