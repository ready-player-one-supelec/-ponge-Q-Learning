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

#%% Q learning
# env = gym.make('Pong-v0' if len(sys.argv)<2 else sys.argv[1])
env = gym.make('CartPole-v1')


reward_global = 0.0

reward_survie = 0.0 #DLC Survie

reward_total = 0.0 

def R(p):
    #return reward_global #à décomenter sans DLC
    return reward_total #DLC survie 

def A_left(obs, score):
    observation, reward, done, info = env.step(0)
    env.render()
    global reward_global
    global reward_survie #DLC Survie
    global reward_total #DLC Survie
    reward_survie = reward_survie + 1 #DLC Survie
    reward_global=reward
    etat = observ_process(observation,obs)
    res = traite(etat,done,reward, score)
    if reward_global < 0: #DLC Survie
        reward_survie = 0  #DLC Survie
        print("reset survie")
    reward_total = reward_global+ 0.8/(1+np.exp((250-reward_survie)/50))  #DLC Survie
    return  res
    
def A_right(obs, score):
    observation, reward, done, info = env.step(1)
    env.render()
    global reward_global
    global reward_survie #DLC Survie
    global reward_total #DLC Survie
    reward_survie = reward_survie + 1 #DLC Survie
    reward_global=reward
    etat = observ_process(observation,obs)
    res = traite(etat,done,reward, score) 
    if reward_global < 0: #DLC Survie
        reward_survie = 0  #DLC Survie
        print("reset survie")
    reward_total = reward_global+ 0.8/(1+np.exp((250-reward_survie)/50))  #DLC Survie
    return res


 
def modify_exp(it,it_max):
    return 0.6*np.exp(-3*it/it_max)+0.2


def chooseDeep(p,A,opt,sess):
    r = np.random.rand()
    if r < opt :
        rr = np.random.randint(len(A)-1)
        return A[rr]
    else:
        fp = sess.run(Predicted_y, {Input_X: [p]})
        res = fp.argmax()
    return A[res]



def sample(D):
    res = []
    for i in range(32):
        elt = D[np.random.randint(len(D))]
        if elt != []:
            res.append(elt)
    return res
            
        


#
#%% fonction principale

def deepQlearning2(A,s0,R,choose,memoire,it,gamma = 0.5):
    scores = []
    inputs = s0 #A voir  --> Implementation ATARI pour s0
    sess.run(tf.global_variables_initializer())
    D = [[] for i in range(memoire)]
    for i in range(it):
        print(" ")
        opt=modify_exp(i,it)
        print("Partie numéro: "+  str(i))
        print("exploration: "+str(round(opt,4)))
        score = 0
        lAS = [s0]
        s = s0
        r = 0
        while np.abs(r) < 0.9 : # Etat final
            a = choose(s,A,opt,sess) #Modify permet de faire evoluer opt par exemple si opt = epsilon on peut le faire décroitre... 
            ss = a(s, score) #ici on fait l'action --> implementation ATARI
            lAS = lAS + [a,ss]
            r = R(ss)
            if [] in D:
                D[D.index([])]=np.array([s,a,r,ss])#si la mémoire n'est pas pleine, on ajoute en mémoire
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
        scores.append(score)
        score = 0
    return("fi")



#%% gym et pretraitement

render = True
env = gym.make('CartPole-v1')
env.reset()
# if render: env.render()

reward_global = 0.0

reward_survie = 0.0 #DLC Survie

reward_total = 0.0


def init_game():
#Initialisation du jeu : les 20 premières frames ne servent à rien
    for _ in range(1):
        # if render: env.render()
        observation, reward, done, info = env.step(0) # take no action (0 is up, 1 is down)
    #     
    # env.reset()
    # env.close()
    pre = observ_process(observation)
    state0 = observ_process(observation,pre)
    return state0

def observ_process(observation, obs = [0,0,0,0]): #crops the image and creates feature vector
    return observation

def traite(etat,done,reward, score):
    # time.sleep(0.01)
    global reward_global
    score += 1
    if done :
        print("score:", score)
        with open('scores.txt', 'a') as file:
            file.write(('\n' + str(score))) 
        env.reset()
    elif reward != 0 :
        toobs, reward, done, info = env.step(0)
        #env.render()
        if done :
            env.reset()
        observation, reward, done, info = env.step(1)
        #env.render()
        if done :
            env.reset()
        obs = observ_process(toobs,toobs)  
        etat = observ_process(observation,obs)
        return traite(etat,done,reward, score)
    else:
        return etat



#%% entrainement
def deep_game(state0,it):
    A = [A_left,A_right]
    s0 = state0
    memoire = 1000
    deepQlearning2(A,s0,R,chooseDeep,memoire,it)
    return('ni')

#%% partie a executer
state0 = init_game()
print(state0)
deep_game(state0,30000)
saver.save(sess, 'my_test_model')

#%% test une fois entrainé (NON UTILISE)
    
def frontprop_deep(A,s0,choose,opt,sess): 
    s = s0
    lS = [s0]
    lA = []
    rs = 0 
    while rs == 0: 
        mouvs = A
        #mouvs = []
        #for a in A: #Cette étape peut rallonger fortement le programme il faudra réflechir à l'enlever 
        #    if a(R,Q,A,s) != -1:
        #        mouvs.append(a)
        aa = choose(s,mouvs,opt,sess)
        lA.append(aa)
        s = aa(s)
        lS.append(s)
        try:
            rs = R(s)
        except KeyError:
            rs = 0
    return [lS,lA]

def test(sess):
    A = [A_left,A_right]
    lss=[]
    for i in range(50):
        state0=init_game()
        lss.append(frontprop_deep(A,state0,chooseDeepPong,0,sess)[0][-1])
    return lss
                

