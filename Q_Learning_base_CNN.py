from prepro_CNN import *
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
model.add(tf.keras.layers.Conv2D(6, 6, input_shape=(80,80,1), strides=(1,1), use_bias=True, padding='same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, input_shape=(6400,1), use_bias=True ))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(2, input_shape=(1,100), use_bias=True ))
model.compile(optimizer='rmsprop',
              loss='mse')



model.load_weights("weights_CNN")
f = open("scorelogs_CNN.txt", 'w')

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
        return (prepro(observation1) - prepro(observation2), action)

    else:
        #choix par le réseau
        state_without_move = state[0].copy()

        up_down_rewards = model.predict(np.array([state_without_move]))[0]

        if up_down_rewards[0].any() > up_down_rewards[1].any() :
            observation1, reward1, done1, _ = env.step(2)
            observation2, reward2, done2, _ = env.step(2)
            action = 0

        else:
            observation1, reward1, done1, _ = env.step(5)
            observation2, reward2, done2, _ = env.step(5)
            action = 1

        global_reward = global_reward + reward1 + reward2
        return ( prepro(observation1) - prepro(observation2) , action)




s=init_game()
observation, reward, done, info = env.step(2)
s = (s - prepro(observation),0.0)
for k in range(it+1):
    states=[]
    while global_reward == 0:
        s = choose_action(s)
        states = states + [s]
            
    if global_reward==-1:
        #entrainer le réseau pour la dernière action négative
        print("Lose")
        etat = states[-1]
        action = etat[1]
        up_down_rewards = model.predict(np.array([etat[0]]))
        
        if action == 0:
            model.fit(np.array([etat[0]]),np.array([[0.0,up_down_rewards[0][1]]]),batch_size=1,epochs=1)

        else:
            model.fit(np.array([etat[0]]),np.array([[up_down_rewards[0][0],0.0]]),batch_size=1,epochs=1)

        

        #si les deux états (haut et bas) vont vers un même reward négatif (<keep_out_threshold), le chemin précédent est mauvais

        N = len(states)
        isWrongPath = True
        l = N-1

        state_without_move = list(states[l])[0]

        up_down_rewards = model.predict(np.array([state_without_move]))

        if (up_down_rewards.any() > keep_out_threshold):
            isWrongPath = False

        while(isWrongPath):
            
            state_without_move = list(states[l][0])
            up_down_rewards = model.predict(np.array([state_without_move]))
            previous_state = np.array([states[l-1][0]])
            previous_action = states[l-1][1]
            previous_action_prediction = model.predict(previous_state)[0]

            if (up_down_rewards.any() > keep_out_threshold):
                isWrongPath = False

            elif previous_action == 0:
                model.fit(previous_state,np.array([[0.0, previous_action_prediction[1]]]),batch_size=1,epochs=1)
                #print("ignoring path - layer " + str(l) + " out of " + str(N))

            else:
                model.fit(previous_state,np.array([[previous_action_prediction[0],0.0]]),batch_size=1,epochs=1)
            
            l = l-1

            if l == 1:
                isWrongPath = False
            

        scores[1] += 1
        global_reward = 0
        

    else:
        #entrainer le réseau pour toutes les actions positives
        print("Win")
        for k in range(1,len(states)):
            state_without_move = list(states[len(states)-k][0])
            up_down_rewards = model.predict(np.array([state_without_move]))[0]
            action = states[len(states)-k][0]
            max_etat_suivant=max(list(up_down_rewards))

            if action.all() == 0:
                new_reward = (1-learning_factor)*model.predict([[states[len(states)-k-1][0]]])[0][0]+learning_factor*(int(k==1)+gamma*max_etat_suivant)
            else:
                new_reward = (1-learning_factor)*model.predict([[states[len(states)-k-1][0]]])[0][1]+learning_factor*(int(k==1)+gamma*max_etat_suivant)

            new_reward = max(new_reward.all(),0.2)
            #new_reward = 1.0

            if action.all() == 0:
                model.fit([[states[len(states)-k-1][0]]],np.array([[new_reward, up_down_rewards[1]]]),batch_size=1,epochs=1)
            else:
                model.fit([[states[len(states)-k-1][0]]],np.array([[up_down_rewards[0], new_reward]]),batch_size=1,epochs=1)

        scores[0] += 1
        global_reward = 0
        
    
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
        s = (s - prepro(observation), 0.0)
    
    if not(k%100):
        model.save_weights("weights_CNN")
    
    
