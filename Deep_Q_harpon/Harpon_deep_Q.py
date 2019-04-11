#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:23:09 2019

@author: odwin
"""
import random as rd 
import numpy as np
import matplotlib.pyplot as plt 
import time


#%% Deep Q Learning 

import Perceptron_Q_Learning as per 

def batch_training(L_inputs,L_th_outputs,reseau,weights,bias,rate,iterations,activation = per.tanh,derivee = per.dtanh): 
    for k in range(iterations):
        delta_weight = [weights[k]*0 for k in range(len(weights))]
        delta_bias = [bias[k]*0 for k in range(len(bias))]
        for data in range(len(L_inputs)):
            gw,gb,cost = per.backprop(L_inputs[data],L_th_outputs[data],reseau,weights,bias,activation,derivee)
            for col in range(len(gw)):
                delta_weight[col] += gw[col]*rate/len(L_inputs)
                delta_bias[col] += gb[col]*rate/len(L_inputs)
        for col in range(len(weights)):
            weights[col] += -delta_weight[col]
            bias[col] += -delta_bias[col]  
    return weights,bias


def sample(D,Dv):
    res = []
    for i in range(32):
        elt = D[rd.randint(0,len(D)-1)]
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
    D[rd.randint(0,len(D)-1)] = elt
    return D

# Alors 
# A est l'ensemble des actions possible (fonctions)
# s0 est l'état de départ 
# choose est la fonction de choix qui prends en entrée s R qw QB A et opt opt étant ce que l'on veut
# R est les resutlats. C'est une fonction : état s --> r(s) Attention il faut faire des try Error a chaque utilisation 
# memoire est la taille de la liste D  de mémoire des arcs on peut la prendre de l'ordre de 1 milion pour un jeu complexe. 
# it est le nombre d'apprentissages (arcs --> echec ou reussite) que l'on réalise
# neural_it est le nombre d'itérations que fait le réseau de neurones a chaque apprentissage
# reseau est le preceprton voulu sans l'entrée  ni la sortie !!!!!!
# Tlimest le temps maximum d'un arc ( pour eviter qu'il tourne en boucle ) attention si le jeu tourne encore apres Tlim alors il va se passer un truc louche 
# phi est la fonction de pré-traitement (s,R,(QW,QB),A,opt)

def deepQlearning(A,s0,R,choose,memoire,it,neural_it,reseau,gamma = 0.5,rate = 0.001,opt = 0,modify = lambda x: x,QW = [],QB = []):
    reseau.append(len(A))
    inputs = s0 #A voir  --> Implementation ATARI pour s0
    if QW == []:
        (QW,QB) = per.random_w_b(inputs,reseau)
    D = [[] for i in range(memoire)]
    Dv = [[] for i in range(memoire)] #DLC ajout du vecteur de victoire 
    for i in range(it):
        print(" ")
        print("Partie numéro: "+  str(i))
        print("exploration: "+str(round(opt,3)))
        lAS = [s0]
        s = s0
        r = 0 
        while np.abs(r) < 0.9 : # Etat final 
            opt = modify(opt) #opt peut etre tout les arguments suplémentaires odnt on a besoin pour le choix 
            a = choose(s,R,(QW,QB),reseau,A,opt) #Modify permet de faire evoluer opt par exemple si opt = epsilon on peut le faire décroitre... 
            ss = a(R,(QW,QB),A,s) #ici on fait l'action --> implementation ATARI
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
                thOutput[k] = per.front_prop(sk,reseau,QW,QB,per.tanh)[-1] #TODO: autre reseau 
                #Parmi les a' on en connait un c'est le a qui se trouve à batch[i][1] et il va falloir calculer son théorical output
                if abs(rk) >= 1: #Etat final 
                    thAk = rk #On calcule l'output théorique
                else:
                    maxk = max(per.front_prop(ssk,reseau,QW,QB,per.tanh)[-1]) #TODO: autre reseau
                    thAk = rk + gamma*maxk #On calcule l'output théorique
                thOutput[k][A.index(ak)] = thAk #On le place au bon endroit !!!!!
#                if rk != 0:
#                    print("")
#                    print(sk)
#                    print(rk)
#                    print(thAk)
#                    print(ssk)
#                    print("")
            #On a fini, plus qu'a entrainer !
            (QW,QB) = batch_training(inputs,thOutput,reseau,QW,QB,rate,neural_it) #TODO: autre reseau
            s = ss
    print(reseau)
    return QW,QB
            


        
        
        
#%% Jeu des batons 

def deepBatons(vitesse = 0.001):
    A = [un_deep,deux_deep]
    s0 = [1 for i in range(11)]
    memoire = 5000
    neural_it = 1
    reseau = [16,8]
    it = 30000
    QW,QB = deepQlearning(A,s0,R,chooseDeepBaton,memoire,it,neural_it,reseau,gamma = 0.6,rate = vitesse,opt = 1,modify = lambda x: (10000*x+0.0)/10001)
    return QW,QB

def testBatons(W,B):
    A = [un_deep,deux_deep]
    s0 = [1 for i in range(11)]
    reseau = [16,8,2]
    res = 0 
    it = 100
    for i in range(it):
        ss = frontprop_deep(A,s0,R,(W,B),reseau,chooseDeepBaton,0)[0][-1]
        if reward_baton == 1:
            res += 1
    res = round(res/it*100)
    print("succes : " + str(res) + " %")
    return res

def testBatons2(W,B):
    reseau = [16,8,2]
    p = [1,0,0,0,0,0,0,0,0,0,0]
    res = per.front_prop(p,reseau,W,B,per.tanh)[-1]
    print( str(p) + "  " + str(res) + "b")
    p = [1,1,0,0,0,0,0,0,0,0,0]
    res = per.front_prop(p,reseau,W,B,per.tanh)[-1]
    print( str(p) + "  " + str(res) + "1")
    p = [1,1,1,0,0,0,0,0,0,0,0]
    res = per.front_prop(p,reseau,W,B,per.tanh)[-1]
    print( str(p) + "  " + str(res) + "2")
    p = [1,1,1,1,0,0,0,0,0,0,0]
    res = per.front_prop(p,reseau,W,B,per.tanh)[-1]
    print( str(p) + "  " + str(res) + "b")
    p = [1,1,1,1,1,0,0,0,0,0,0]
    res = per.front_prop(p,reseau,W,B,per.tanh)[-1]
    print( str(p) + "  " + str(res) + "1")
    p = [1,1,1,1,1,1,0,0,0,0,0]
    res = per.front_prop(p,reseau,W,B,per.tanh)[-1]
    print( str(p) + "  " + str(res) + "2")
    p = [1,1,1,1,1,1,1,0,0,0,0]
    res = per.front_prop(p,reseau,W,B,per.tanh)[-1]
    print( str(p) + "  " + str(res) + "b")
    p = [1,1,1,1,1,1,1,1,0,0,0]
    res = per.front_prop(p,reseau,W,B,per.tanh)[-1]
    print( str(p) + "  " + str(res) + "1")
    p = [1,1,1,1,1,1,1,1,1,0,0]
    res = per.front_prop(p,reseau,W,B,per.tanh)[-1]
    print( str(p) + "  " + str(res) + "2")
    p = [1,1,1,1,1,1,1,1,1,1,0]
    res = per.front_prop(p,reseau,W,B,per.tanh)[-1]
    print( str(p) + "  " + str(res) + "b")
    p = [1,1,1,1,1,1,1,1,1,1,1]
    res = per.front_prop(p,reseau,W,B,per.tanh)[-1]
    print( str(p) + "  " + str(res) + "1")


reward_baton = 0      

reward_list = []    

def R(p):
    return reward_baton



def chooseDeepBaton(p,R,Q,reseau,A,opt):
    (QW,QB) = Q
    r = rd.random()
    if r < opt :
        rr = rd.randint(0,len(A)-1)
        return A[rr]
    else:
        fp = per.front_prop(p,reseau,QW,QB,per.tanh)[-1] 
        res = fp.argmax()
        return A[res]
        
   
def deux_deep(R,Q,A,p):
    global reward_baton
    s= sum(p)
    if s == 1:
        print("[u,0,0,0,0,0,0,0,0,0,0] DEFAITE 3")
        reward_baton = -10
        reward_list.append(reward_baton)
        return [1,1,1,1,1,1,1,1,1,1,1]
    if s == 2:
        print("[u,u,0,0,0,0,0,0,0,0,0] DEFAITE 2")
        reward_baton = -10
        reward_list.append(reward_baton)
        return [1,1,1,1,1,1,1,1,1,1,1]
    if s == 3:
        print("[e,u,u,0,0,0,0,0,0,0,0] VICTOIRE 2")
        reward_baton = 10
        reward_list.append(reward_baton)
        return [1,1,1,1,1,1,1,1,1,1,1]
    if s == 4:
        #print("[1,e,u,u,0,0,0,0,0,0,0]")
        reward_baton = 0
        return [1,0,0,0,0,0,0,0,0,0,0]
    if s == 5:
        #print("[1,e,e,u,u,0,0,0,0,0,0]")
        reward_baton = 0
        return [1,0,0,0,0,0,0,0,0,0,0]
    if s == 6:
        #print("[1,1,1,e,u,u,0,0,0,0,0]")
        reward_baton = 0
        return [1,1,1,0,0,0,0,0,0,0,0]
    if s == 7:
        #print("[1,1,1,1,e,u,u,0,0,0,0]")
        reward_baton = 0
        return [1,1,1,1,0,0,0,0,0,0,0]
    if s == 8:
        #print("[1,1,1,1,e,e,u,u,0,0,0]")
        reward_baton = 0
        return [1,1,1,1,0,0,0,0,0,0,0]
    if s == 9:
        #print("[1,1,1,1,1,1,e,u,u,0,0]")
        reward_baton = 0
        return [1,1,1,1,1,1,0,0,0,0,0]
    if s == 10:
        #print("[1,1,1,1,1,1,1,e,u,u,0]")
        reward_baton = 0
        return [1,1,1,1,1,1,1,0,0,0,0]
    if s == 11:
        #print("[1,1,1,1,1,1,1,e,e,u,u]")
        reward_baton = 0
        return [1,1,1,1,1,1,1,0,0,0,0]
        
def un_deep(R,Q,A,p):
    global reward_baton
    s= sum(p)
    if s == 1:
        print("[u,0,0,0,0,0,0,0,0,0,0] DEFAITE 1")
        reward_baton = -10
        reward_list.append(reward_baton)
        return [1,1,1,1,1,1,1,1,1,1,1]
    if s == 2:
        print("[e,u,0,0,0,0,0,0,0,0,0] VICTOIRE 1")
        reward_baton = 10
        reward_list.append(reward_baton)
        return [1,1,1,1,1,1,1,1,1,1,1]
    if s == 3:
        #print("[1,e,u,0,0,0,0,0,0,0,0]")
        reward_baton = 0
        return [1,0,0,0,0,0,0,0,0,0,0]
    if s == 4:
        #print("[1,e,e,u,0,0,0,0,0,0,0]")
        reward_baton = 0
        return [1,0,0,0,0,0,0,0,0,0,0]
    if s == 5:
        #print("[1,1,1,e,u,0,0,0,0,0,0]")
        reward_baton = 0
        return [1,1,1,0,0,0,0,0,0,0,0]
    if s == 6:
        #print("[1,1,1,1,e,u,0,0,0,0,0]")
        reward_baton = 0
        return [1,1,1,1,0,0,0,0,0,0,0]
    if s == 7:
        #print("[1,1,1,1,e,e,u,0,0,0,0]")
        reward_baton = 0
        return [1,1,1,1,0,0,0,0,0,0,0]
    if s == 8:
        #print("[1,1,1,1,1,e,e,u,0,0,0]")
        reward_baton = 0
        return [1,1,1,1,1,0,0,0,0,0,0]
    if s == 9:
        #print("[1,1,1,1,1,1,1,e,u,0,0]")
        reward_baton = 0
        return [1,1,1,1,1,1,1,0,0,0,0]
    if s == 10:
        #print("[1,1,1,1,1,1,1,e,e,u,0]")
        reward_baton = 0
        return [1,1,1,1,1,1,1,0,0,0,0]
    if s == 11:
        #print("[1,1,1,1,1,1,1,1,e,e,u]")
        reward_baton = 0
        return [1,1,1,1,1,1,1,1,0,0,0]
    



def Batons():
    l = [0.1,0.01,0.0075,0.005,0.0025,0.001,0.00075,0.0005,0.00025,0.0001]
    resliste = []
    for vitesse in l:
        W,B = deepBatons(vitesse)  
        reward = [sum([(reward_list[100*j +i] == 10) for i in range(100)])/100 for j in range(len(reward_list )//100)]
        resliste.append(reward)
    return resliste

res = Batons()
#%%
    
def frontprop_deep(A,s0,R,Q,reseau,choose,opt = 1): 
    s = s0
    lS = [s0]
    lA = []
    rs = 0 
    while rs == 0: 
        aa = choose(s,R,Q,reseau,A,opt)
        lA.append(aa)
        s = aa(R,Q,A,s)
        lS.append(s)
        rs = R(s)
    return [lS,lA]

        
#%%
#reseau = [8,4,2]
#print(1)
#print(per.front_prop([1,0,0,0,0,0,0,0,0,0,0],reseau,A,B,per.tanh)[-1])
#print(2)
#print(per.front_prop([1,1,0,0,0,0,0,0,0,0,0],reseau,A,B,per.tanh)[-1])
#print(3)
#print(per.front_prop([1,1,1,0,0,0,0,0,0,0,0],reseau,A,B,per.tanh)[-1])
#print(5)
#print(per.front_prop([1,1,1,1,1,0,0,0,0,0,0],reseau,A,B,per.tanh)[-1])
