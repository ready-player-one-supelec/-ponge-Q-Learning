#! /usr/bin/python3
# -*- coding:utf-8 -*-

from interface import *
import random

with Game(display = 1, returnFeatures = 1) as game :
    continuer = True
    while continuer :
        m = random.randint(0, 100)
        m = int(m >= 97)
        continuer, image, reward = game.game_step(m)
