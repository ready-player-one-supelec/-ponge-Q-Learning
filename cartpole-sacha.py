#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:30:33 2019

@author: sacha
"""

import tensorflow as tf
import numpy as np
import gym


sess=tf.InteractiveSession()

## Defining various initialization parameters for 784-512-256-10 MLP model
Num_classes = 4
Num_output = 2
Num_layers_0 = 32
Num_layers_1 = 32
starter_learning_rate = 0.001

# Placeholders for the input data
Input_X = tf.placeholder('float32',shape =(None,Num_classes),name="Input_X")
Input_Y = tf.placeholder('float32',shape = (None,Num_output),name='Input_Y')
## for dropout layer


#initialising biases and weights
## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
Weights_0 = tf.Variable(tf.random_normal([Num_classes,Num_layers_0]))
Bias_0 = tf.Variable(tf.random_normal([Num_layers_0]))
Weights_1 = tf.Variable(tf.random_normal([Num_layers_0,Num_layers_1], stddev=(1/tf.sqrt(float(Num_layers_0)))))
Bias_1 = tf.Variable(tf.random_normal([Num_layers_1]))
Weights_2 = tf.Variable(tf.random_normal([Num_layers_1,Num_output], stddev=(1/tf.sqrt(float(Num_layers_1)))))
Bias_2 = tf.Variable(tf.random_normal([Num_output]))

## Initializing network
Hidden_output_0 = tf.nn.tanh(tf.matmul(Input_X,Weights_0)+Bias_0)
Hidden_output_1 = tf.nn.tanh(tf.matmul(Hidden_output_0,Weights_1)+Bias_1)
Predicted_y = tf.nn.tanh(tf.matmul(Hidden_output_1,Weights_2) + Bias_2)

## Defining the loss function (quadratic)
Loss=tf.losses.mean_squared_error(Input_Y,Predicted_y)

## Variable learning rate
#learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 1000, 0.95, staircase=True)

## Adam optimzer for finding the right weight
Optimizer = tf.train.AdamOptimizer(starter_learning_rate).minimize(Loss,var_list=[Weights_0,Weights_1,Weights_2,Bias_0,Bias_1,Bias_2])

## Metrics definition
#correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predicted_y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
Accuracy = 1-Loss

saver=tf.train.Saver()

sess.run(tf.global_variables_initializer())

#%% Q learning
# env = gym.make('Pong-v0' if len(sys.argv)<2 else sys.argv[1])
env = gym.make('CartPole-v1')
writebool = False

score=0

def R(p):
    global score
    #return reward_global #à décomenter sans DLC
    return bool(score)*2-1 #DLC survie 

def A_left(obs):
    observation, reward, done, info = env.step(0)
    global score
    traite(done)
    if score==0: #DLC Survie
    return  observation
    
def A_right(obs):
    observation, reward, done, info = env.step(1)
    global score
    traite(done)
    if score==0: #DLC Survie
    return  observation
    

 
def exploration_exp(it,it_max):
    return 0.6*np.exp(-3*it/it_max)+0.2

def exploration_lin(it,it_max):
    return 1-it/it_max

def chooseDeep(p,A,opt,sess):
    r = np.random.rand()
    if r < opt :
        rr = np.random.randint(len(A))
        return A[rr]
    fp = sess.run(Predicted_y, {Input_X: [p]})
    return A[fp.argmax()]


def sample(D):
    res = []
    for i in range(32):
        elt = D[np.random.randint(len(D))]
        if elt != []:
            res.append(elt)
    return res           

#
#%% fonction principale

def deepQlearning2(A,R,choose,memoire,it,gamma = 0.995):
    global writebool
    memorymark=0
    D = [[] for i in range(memoire)]
    for i in range(it):
        print(" ")
        opt=exploration_lin(i,it)
        print("Partie numéro: "+  str(i))
        print("exploration: "+str(round(opt,4)))
        env.reset()
        s0=init_game()
        s = s0
        r = R(s)
        #training game
        writebool = False
        while r>0  : # Etat final
            a = choose(s,A,opt,sess)# choix de l'action. 
            ss = a(s) #ici on fait l'action --> implementation ATARI
            r = R(ss)
            if memorymark<memoire:
                D[memorymark]=np.array([s,a,r,ss])#si la mémoire n'est pas pleine, on ajoute en mémoire
                memorymark+=1
            else:
                D[np.random.randint(len(D)-1)] = np.array([s,a,r,ss])#si elle est pleine on remplace aléatoirement
            batch = sample(D) #On eslectione des arcs pour apprendre
            thOutput = np.array([[0 for jj in range(len(A))] for ii in range(len(batch))])
            inputs = np.array([s for ii in range(len(batch))]) #On initialise les inputs qui vont etre les s du batch ( ici s sert uniquement a donner la taille )
            thAk = 0 #Initialisation de la variable 
            for k in range(len(batch)): # On doit d'abbord calculer les sorties que l'on connait 
                sk = batch[k][0]
                inputs[k] = sk 
                ak = batch[k][1]
                rk= batch[k][2]
                ssk = batch[k][3] #On récupère les données du batch 
                #On va céer les valeurs de Q(s,a') pour les a' que l'on ne connait pas 
                thOutput[k] = sess.run(Predicted_y, {Input_X:[sk]}) #TODO: autre reseau 
                #Parmi les a' on en connait un c'est le a qui se trouve à batch[i][1] et il va falloir calculer son théorical output
                if abs(rk) >= 1: #Etat final 
                    thAk = rk #On calcule l'output théorique
                else:
                    maxk = max(sess.run(Predicted_y, {Input_X:[ssk]})[-1]) #TODO: autre reseau
                    thAk = rk + gamma*maxk #On calcule l'output théorique
                thOutput[k][A.index(ak)] = thAk #On le place au bon endroit !!!!!
            if len(batch)!=0:
                sess.run(Optimizer, {Input_X: inputs, Input_Y: thOutput}) #TODO: autre reseau
            s = ss
        #test game
        env.reset()
        s0=init_game()
        s=s0
        r=R(s)
        opt=0
        writebool = True
        print("test run:")
        while r>0  : # Etat final
            a = choose(s,A,opt,sess) # choix de l'action
            ss = a(s) #ici on fait l'action --> implementation ATARI
            r = R(ss)
            s = ss
    return("fi")



#%% gym et pretraitement

render = False
env = gym.make('CartPole-v1')
env.reset()
# if render: env.render()

def init_game():
#Initialisation du jeu : les 20 premières frames ne servent à rien
    observation, reward, done, info = env.step(0) # take no action (0 is left, 1 is right)     
    traite(done)
    state0 = observation
    return state0


def traite(done): #décide si il faut sauvegarder les résultats, augmente le score
    # time.sleep(0.01)
    global score
    score += 1
    if done :
        print("score final:",score)
        if writebool:
            with open('scores.txt', 'a') as file:
                file.write(('\n' + str(score))) 
        score=0
    return None


#%% entrainement
def deep_game(state0,it):
    A = [A_left,A_right]
    memoire = 10000
    deepQlearning2(A,R,chooseDeep,memoire,it)
    return('ni')

#%% partie a executer
state0 = init_game()
print(state0)
deep_game(state0,50000)
saver.save(sess, 'my_test_model')


