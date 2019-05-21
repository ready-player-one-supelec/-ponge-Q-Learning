from prepro import *
import gym
import time
import numpy as np
import random as rd
import time
import Perceptron_Q_Learning as per 
import Harpon_Q_Learning as q
import tensorflow as tf

from conf import *

env = gym.make('Pong-v0')
env.reset()
if render: env.render()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1000, input_shape=(6401,), use_bias=True ))
model.add(tf.keras.layers.Activation('relu'))
model.compile(optimizer='rmsprop',
              loss='mse')



model.load_weights("weights")
f = open("scorelogs.txt", 'w')

def init_game():
#Initialisation du jeu : les 20 premières frames ne servent à rien
    for _ in range(21):
        if render: env.render()
        observation, reward, done, info = env.step(0) # take no action (2 is up, 5 is down)

    env.reset()
    env.close()
    
    state0 = prepro(observation)
    return state0

learning_factor = 0.8
current_rate = initial_exploration_rate
global_reward = 0
scores = [0,0]
gamma=0.6


def choose_action(state):
    global global_reward

    if rd.random()<current_rate:
        #choix par le random
        action = rd.randint(0,1)
        if action == 0 :
            observation1, reward1, done1, _ = env.step(2)
            observation2, reward2, done2, _ = env.step(2)
        else:
            observation1, reward1, done1, _ = env.step(5)
            observation2, reward2, done2, _ = env.step(5)

        global_reward = global_reward + reward1 + reward2
        return (np.concatenate((prepro(observation1) - prepro(observation2), np.array([action]))))

    else:
        #choix par le réseau
        state_without_move = (list(state)).copy()
        state_without_move = state_without_move[:-1].copy()

        state_move_up = (state_without_move + [0]).copy()
        state_move_down = (state_without_move + [1]).copy()

        up_state = [[state_move_up]]
        down_state = [[state_move_down]]

        up_reward = model.predict(up_state)
        down_reward = model.predict(down_state)

        if up_reward.all() > down_reward.all() :
            observation1, reward1, done1, _ = env.step(2)
            observation2, reward2, done2, _ = env.step(2)
            action = 0
        else:
            observation1, reward1, done1, _ = env.step(5)
            observation2, reward2, done2, _ = env.step(5)
            action = 1

        global_reward = global_reward + reward1 + reward2
        return ( np.concatenate(( prepro(observation1) - prepro(observation2), np.array([action]))))




s=init_game()
observation, reward, done, info = env.step(2)
s = np.concatenate((s - prepro(observation), np.array([0])))
for k in range(it+1):
    states=[]
    while global_reward == 0:
        s = choose_action(s)
        states = states + [s]
            
    if global_reward==-1:
        #entrainer le réseau pour la dernière action négative

        etat = np.array([states[-1]])
        model.fit(etat,[0],batch_size=1,epochs=1)

        #si les deux états (haut et bas) vont vers un même reward négatif (<keep_out_threshold), le chemin précédent est mauvais

        N = len(states)
        isWrongPath = True
        l = N-1

        state_without_move = list(states[l])[:-1]
        up_move = state_without_move + [0]
        down_move = state_without_move + [1]
        valeur_up=model.predict([[up_move]])
        valeur_down=model.predict([[down_move]])

        if (valeur_up.any() >= keep_out_threshold or valeur_down.any() >= keep_out_threshold):
            isWrongPath = False

        while(isWrongPath):
            
            state_without_move = list(states[l])[:-1]
            up_move = state_without_move + [0]
            down_move = state_without_move + [1]
            valeur_up=model.predict([[up_move]])
            valeur_down=model.predict([[down_move]])
            previous_state = np.array([states[l-1]])

            if (valeur_up.any() >= keep_out_threshold or valeur_down.any() >= keep_out_threshold):
                isWrongPath = False

            else:
                model.fit(previous_state,[0],batch_size=1,epochs=1)
                #print("ignoring path - layer " + str(l) + " out of " + str(N))
            
            l = l-1

            if l == 1:
                isWrongPath = False
            



        scores[1] += 1
        global_reward = 0
        print("Lose")

    else:
        #entrainer le réseau pour toutes les actions positives

        for k in range(1,len(states)):
            state_without_move = list(states[len(states)-k])[:-1]
            up_move = state_without_move + [0]
            down_move = state_without_move + [1]
            valeur_up=model.predict([[up_move]])
            valeur_down=model.predict([[down_move]])
            max_etat_suivant=max(valeur_down.all(),valeur_up.all())

            #new_reward = (1-learning_factor)*model.predict([[states[len(states)-k-1]]])+learning_factor*(int(k==1)+gamma*max_etat_suivant)
            #max_reward = max(new_reward.all(),0.2)
            new_reward = 1
            model.fit([[states[len(states)-k-1]]],[new_reward],batch_size=1,epochs=1)

        scores[0] += 1
        global_reward = 0
        print("Win")
    
    print(k)

    current_rate = current_rate * exploration_rate
    print(current_rate)


    if 21 in scores:
        print("Score : " + str(scores), file = f)
        scores = [0,0]
        env.reset()
        env.close()
        s=init_game()
        observation, reward, done, info = env.step(2)
        s = np.concatenate((s - prepro(observation), np.array([0])))
    
    if not(k%100):
        model.save_weights("weights")
    
    
