#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:30:33 2019

@author: sacha
"""

import tensorflow as tf
import numpy as np
import gym
import sys


sess=tf.InteractiveSession()

## Defining various initialization parameters for 784-512-256-10 MLP model
Num_classes = 8
Num_output = 2
Num_layers_0 = 32
Num_layers_1 = 32
starter_learning_rate = 0.001
dropout_prob = 0.5

# Placeholders for the input data
Input_X = tf.placeholder('float32',shape =(None,Num_classes),name="Input_X")
Input_Y = tf.placeholder('float32',shape = (None,Num_output),name='Input_Y')
## for dropout layer
keep_prob = tf.placeholder(tf.float32)


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
Hidden_output_0_0 = tf.nn.dropout(Hidden_output_0, keep_prob)
Hidden_output_1 = tf.nn.tanh(tf.matmul(Hidden_output_0_0,Weights_1)+Bias_1)
Hidden_output_1_1 = tf.nn.dropout(Hidden_output_1, keep_prob)
Predicted_y = tf.nn.tanh(tf.matmul(Hidden_output_1_1,Weights_2) + Bias_2)

## Defining the loss function (quadratic)
Loss=tf.losses.mean_squared_error(Input_Y,Predicted_y)

## Variable learning rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 1000, 0.95, staircase=True)

## Adam optimzer for finding the right weight
Optimizer = tf.train.AdamOptimizer(learning_rate).minimize(Loss,var_list=[Weights_0,Weights_1,Weights_2,Bias_0,Bias_1,Bias_2])

## Metrics definition
#correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predicted_y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
Accuracy = 1-Loss

saver=tf.train.Saver()

#%% Q learning
env = gym.make('Pong-v0' if len(sys.argv)<2 else sys.argv[1])


reward_global = 0.0

reward_survie = 0.0 #DLC Survie

reward_total = 0.0 

def R(p):
    #return reward_global #à décomenter sans DLC
    return reward_total #DLC survie 

def A_up(obs):
    observation, reward, done, info = env.step(2)
    env.render()
    global reward_global
    global reward_survie #DLC Survie
    global reward_total #DLC Survie
    reward_survie = reward_survie + 1 #DLC Survie
    reward_global=reward
    etat = observ_process(observation,obs)
    res = traite(etat,done,reward)
    if reward_global < 0: #DLC Survie
        reward_survie = 0  #DLC Survie
        print("reset survie")
    reward_total = reward_global+ 0.8/(1+np.exp((250-reward_survie)/50))  #DLC Survie
    return  res
    
def A_down(obs):
    observation, reward, done, info = env.step(5)
    env.render()
    global reward_global
    global reward_survie #DLC Survie
    global reward_total #DLC Survie
    reward_survie = reward_survie + 1 #DLC Survie
    reward_global=reward
    etat = observ_process(observation,obs)
    res = traite(etat,done,reward) 
    if reward_global < 0: #DLC Survie
        reward_survie = 0  #DLC Survie
        print("reset survie")
    reward_total = reward_global+ 0.8/(1+np.exp((250-reward_survie)/50))  #DLC Survie
    return res


def modify(x):
    return (100000*x+0.3)/100001

 Def modify_exp(it,it_max):
    Return 0.6*np.exp(-3*it/it_max)+0.2

def chooseDeepPong(p,A,opt,sess):
    r = np.random.rand()
    if r < opt :
        rr = np.random.randint(len(A)-1)
        return A[rr]
    else:
        fp = sess.run(Predicted_y, {Input_X: [p], keep_prob: 1})
        res = fp.argmax()
    return A[res]



def sample(D,Dv):
    res = []
    for i in range(32):
        elt = D[np.random.randint(len(D)-1)]
        if elt != []:
            res.append(elt)
# =============================================================================
#     for i in range(16): #DLC ajout du vecteur de victoire 
#         elt = Dv[rd.randint(0,len(Dv)-1)] #DLC ajout du vecteur de victoire 
#         if elt != []: #DLC ajout du vecteur de victoire 
#             res.append(elt)   #DLC ajout du vecteur de victoire 
# =============================================================================
    return res
            
        

def ajoute(D,elt):
    D[np.random.randint(len(D)-1)] = elt
    return D


def phibase(l): #Dans le cas vraiment basique pas besoin de faire de traitement on retient donc les arcs (s,a,r,ss)
    return l[-1] # apres on pert le corrélation entre les arcs donc c'est plutot mauvais 


#
#%% fonction principale

def deepQlearning2(A,s0,R,choose,memoire,it,gamma = 0.5):
    global sess
    inputs = s0 #A voir  --> Implementation ATARI pour s0
    sess.run(tf.global_variables_initializer())
    D = [[] for i in range(memoire)]
    Dv = [[] for i in range(memoire)] #DLC ajout du vecteur de victoire 
    for i in range(it):
        print(" ")
        print("Partie numéro: "+  str(i))
        print("exploration: "+str(round(opt,4)))
        lAS = [s0]
        s = s0
        r = 0 
        opt = modify_exp(i,it)
        while np.abs(r) < 0.9 : # Etat final
            
            a = choose(s,A,opt,sess) #Modify permet de faire evoluer opt par exemple si opt = epsilon on peut le faire décroitre... 
            ss = a(s) #ici on fait l'action --> implementation ATARI
            lAS = lAS + [a,ss]
            r = R(ss)
# =============================================================================
#             if r == 1 : #DLC ajout du vecteur de victoire
#                 sv = lAS[0] #DLC ajout du vecteur de victoire
#                 for i in range(1,len(lAS)//2): #DLC ajout du vecteur de victoire
#                     ssv = lAS[2*i]  #DLC ajout du vecteur de victoire
#                     av = lAS[2*i-1]  #DLC ajout du vecteur de victoire
#                     rv = 1*(len(lAS)-2*i)/len(lAS)
#                     Dv = ajoute(Dv,np.array([sv,av,rv,ssv]))  #DLC ajout du vecteur de victoire
# =============================================================================
            D  = ajoute(D,np.array([s,a,r,ss])) #!!
            batch = sample(D,Dv) #On eslectione des arcs pour apprendre
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
                thOutput[k] = sess.run(Predicted_y, {Input_X:[sk], keep_prob:1}) #TODO: autre reseau 
                #Parmi les a' on en connait un c'est le a qui se trouve à batch[i][1] et il va falloir calculer son théorical output
                if abs(rk) >= 1: #Etat final 
                    thAk = rk #On calcule l'output théorique
                else:
                    maxk = max(sess.run(Predicted_y, {Input_X:[ssk], keep_prob:1})[-1]) #TODO: autre reseau
                    thAk = rk + gamma*maxk #On calcule l'output théorique
                thOutput[k][A.index(ak)] = thAk #On le place au bon endroit !!!!!
            if len(batch)!=0:
                sess.run(Optimizer, {Input_X: inputs, Input_Y: thOutput, keep_prob:dropout_prob}) #TODO: autre reseau
            s = ss
    #return(sess.run(Weights_0),sess.run(Bias_0),sess.run(Weights_1),sess.run(Bias_1),sess.run(Weights_2),sess.run(Bias_2))
    return("fi")



#%% gym et pretraitement

render = True
env = gym.make('Pong-v0')
env.reset()
# if render: env.render()

reward_global = 0.0

reward_survie = 0.0 #DLC Survie

reward_total = 0.0 


def init_game():
#Initialisation du jeu : les 20 premières frames ne servent à rien
    for _ in range(21):
        # if render: env.render()
        observation, reward, done, info = env.step(0) # take no action (2 is up, 5 is down)
    #     
    # env.reset()
    # env.close()
    pre = observ_process(observation)
    state0 = observ_process(observation,pre)
    return state0

def observ_process(observation, obs = [0,0,0,0,0,0,0,0]): #crops the image and creates feature vector
    observationR=[[0 for k in range(len(observation[0]))] for l in range(34,len(observation)-15)]
    for k in range(34,len(observation)-15):    
        for l in range(len(observation[0])):
            observationR[k-34][l]=observation[k][l][0]
    c = observationR
    d = np.array(c).T
    n = len(d)
    m = len(d[0])
    dd = np.array([[0 for j in range(m-1)] for i in range(n)])
    for i in range(n):
        dd[i] = d[i][0:m-1]
    m = len(dd[0])
    n = len(dd)
    d18 = np.array([dd[18][m-i-1] for i in range(m)])
    e4 = m - dd[18].argmax()
    e3 = d18.argmax()
    d140 = np.array([dd[140][m-i-1] for i in range(m)])
    e2 = m - dd[140].argmin()
    e1 = d140.argmin()
    uf = True
    e5 = 0
    e6 = 0
    for i in range(20,140):
        if dd[i].max() != 144 and uf :
            e6 = m - dd[i].argmax()
            e5 = 160 - i
            uf = False
    e7 = obs[4]
    e8 = obs[5]
    e1 = e1/m-0.5
    e2 = e2/m-0.5
    e3 = e3/m-0.5
    e4 = e4/m-0.5
    e5 = e5/m-0.5
    e6 = e6/m-0.5
    return [e1,e2,e3,e4,e5,e6,e7,e8]

def traite(etat,done,reward):
    global reward_global
    if done :
        env.reset()
    if etat[4] == -0.5 and etat[5] == -0.5 :
        toobs, reward, done, info = env.step(2)
        #env.render()
        if done :
            env.reset()
        if reward != 0 :
            reward_global=reward
        observation, reward, done, info = env.step(5)
        #env.render()
        if done :
            env.reset()
        if reward != 0 :
            reward_global=reward
        obs = observ_process(toobs,toobs)
        etat = observ_process(observation,obs)
        return traite(etat,done,reward)
    elif reward != 0 :
        toobs, reward, done, info = env.step(2)
        #env.render()
        if done :
            env.reset()
        observation, reward, done, info = env.step(5)
        #env.render()
        if done :
            env.reset()
        obs = observ_process(toobs,toobs)  
        etat = observ_process(observation,obs)
        return traite(etat,done,reward)
    else:
        return etat



#%% entrainement
def deep_pong(state0,it):
    A = [A_up,A_down]
    s0 = state0
    memoire = 1000
    deepQlearning2(A,s0,R,chooseDeepPong,memoire,it)
    return('ni')

#%% partie a executer

state0 = init_game()
print(state0)
deep_pong(state0,30000)
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
    A = [A_up,A_down]
    lss=[]
    for i in range(50):
        state0=init_game()
        lss.append(frontprop_deep(A,state0,chooseDeepPong,0,sess)[0][-1])
    return lss
                


#
#
#
#    
## for _ in range(1000):
##     if rd.randint(1,2)==1:
##         A_up(observation, reward, done, info)
##         print("up")
##     else:
##         A_down(observation, reward, done, info)
##         print("down")
#
#env.reset()
#env.close()
