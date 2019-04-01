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

def phibase(l): #Dans le cas vraiment basique pas besoin de faire de traitement on retient donc les arcs (s,a,r,ss)
    return l[-1] # apres on pert le corrélation entre les arcs donc c'est plutot mauvais 

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

def fini_base(R,opt,opt2):
    return (np.abs(R) > 0.9) 

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

def deepQlearning(A,s0,R,choose,memoire,it,neural_it,reseau,fini = fini_base,phi = phibase,gamma = 0.5,rate = 0.001,opt = 0,modify = lambda x: x,QW = [],QB = []):
    Tlim = 10e9 
    reseau.append(len(A))
    inputs = s0 #A voir  --> Implementation ATARI pour s0
    if QW == []:
        (QW,QB) = per.random_w_b(inputs,reseau)
    D = [[] for i in range(memoire)]
    Dv = [[] for i in range(memoire)] #DLC ajout du vecteur de victoire 
    for i in range(it):
        print("Partie numéro: "+  str(i))
        print("exploration: "+str(round(opt[0],3)))
        lAS = [s0]
        s = s0
        p = phi(lAS)
        r = 0
        j = 0 #Evite une boucle inifnie mais doit etre élevé pour ne pas bloquer le jeu 
        while j<Tlim and not(fini(r)) : # Etat final 
            j += 1
            opt = modify(opt) #opt peut etre tout les arguments suplémentaires odnt on a besoin pour le choix 
            a = choose(p,R,(QW,QB),reseau,A,opt) #Modify permet de faire evoluer opt par exemple si opt = epsilon on peut le faire décroitre... 
            ss = a(R,(QW,QB),A,s,opt) #ici on fait l'action --> implementation ATARI
            lAS = lAS + [a,ss]
            pp = phi(lAS)
            try:
                r = R(ss)
            except KeyError:
                r = 0
# =============================================================================
#             if r == 1 : #DLC ajout du vecteur de victoire
#                 sv = lAS[0] #DLC ajout du vecteur de victoire
#                 for i in range(1,len(lAS)//2): #DLC ajout du vecteur de victoire
#                     ssv = lAS[2*i]  #DLC ajout du vecteur de victoire
#                     av = lAS[2*i-1]  #DLC ajout du vecteur de victoire
#                     rv = 1*(len(lAS)-2*i)/len(lAS) #DLC ajout du vecteur de victoire
#                     Dv = ajoute(Dv,np.array([sv,av,rv,ssv]))  #DLC ajout du vecteur de victoire
# =============================================================================
            D  = ajoute(D,np.array([p,a,r,pp])) #!!
            batch = sample(D,Dv) #On eslectione des arcs pour apprendre 
            inputs = [batch[i][0] for i in range(len(batch))] #La seule partie qui nous interesse pour les inputs c'est l'arrivée 
            y = [0 for k in range(len(batch))] # Calcul de Theorical output Initialisation
            for k in range(len(batch)): # On doit d'abbord calculer les sorties que l'on connait 
                #sk = batch[k][0]
                #ak = batch[k][1]
                rk= batch[k][2]
                ssk = batch[k][3] #On récupère 
                if abs(rk) >= 1: #Etat final 
                    y[k] = rk #On calcule l'output théorique
                else:
                    maxk = max(per.front_prop(ssk,reseau,QW,QB,per.tanh)[-1])
                    y[k] = rk + gamma*maxk #On calcule l'output théorique
            thOutput = np.array([[0 for jj in range(len(A))] for ii in range(len(inputs))]) #On crée cela pour avoir les valeurs de Q(s,a') pour les a' que l'on ne connait pas 
            for ii in range(len(inputs)):
                thOutput[ii] = per.front_prop(inputs[ii],reseau,QW,QB,per.tanh)[-1]
            for k in range(len(y)) : #On change la forme de y c'est pas l'ideal mais ca a été fait comme ca 
                thOutput[k][A.index(a)] = y[k]
            (QW,QB) = batch_training(inputs,thOutput,reseau,QW,QB,rate,neural_it)
            #print("temps 1:" + str(round(abs(100*(ta-tb)))))
            #print("temps 2:" + str(round(abs(100*(tc-tb)))))
            #print("temps 3:" + str(round(abs(100*(tc-td)))))
            #print("")
            s = ss
            p = pp
    print(reseau)
    return QW,QB
            


        
        
        
#%% Jeu des batons 

def deepBatons(it = 50):
    A = [un_deep,deux_deep]
    s0 = [1 for i in range(11)]
    memoire = 1000
    neural_it = 10
    reseau = [8,4]
    QW,QB = deepQlearning(A,s0,R,chooseDeepBaton,memoire,it,neural_it,reseau,Tlim = 10e9,phi = phibase,gamma = 0.6,rate = 0.0005,opt = 0.8,modify = lambda x: (100*x+0.1)/101)
    res = 0
    for i in range(1000):
        ss = frontprop_deep(A,s0,R,(QW,QB),reseau,chooseDeepBaton,0)[0][-1]
        if ss[0] == 2:
            res += 1
    res = res /1000
    print(res)
    return QW,QB,res

        

def R(p):
    if p[0] == -1: return -1
    elif p[0] == 2: return 1
    else : return 0



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
    s= sum(p)
    if s-2 < 1:
        return [-1 for i in range(11)]
    else:
        e = ennemi(R,Q,A,s)
        res = s-2-e
        if res < 1:
            return [2 for i in range(11)]
        else:
            pp = [0 for i in range(11)]
            for j in range(res):
                pp[j] = 1
            return pp
        
def un_deep(R,Q,A,p):
    s= sum(p)
    if s-1 < 1:
        return [-1 for i in range(11)]
    else:
        e = ennemi(R,Q,A,s)
        res = s-1
        -e
        if res < 1:
            return [2 for i in range(11)]
        else:
            pp = [0 for i in range(11)]
            for j in range(res):
                pp[j] = 1
            return pp
    
def ennemi(R,Q,A,s): #Stategie gagnante 
    if s%3 == 1: return rd.randint(1,2)
    if s%3 == 0: return 2
    if s%3 == 2: return 1
    
#%%
    
def frontprop_deep(A,s0,R,Q,reseau,choose,opt = 1): 
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
        aa = choose(s,R,Q,reseau,mouvs,opt)
        lA.append(aa)
        s = aa(R,Q,A,s,opt)
        lS.append(s)
        try:
            rs = R(s)
        except KeyError:
            rs = 0
    return [lS,lA]

def courbeBaton(nb = 10):
    res = []
    for i in range(1,nb):
        A,B,taux = deepBatons(i*50)
        res.append(taux)
    plt.plot(res)
    return res
        

#A,B,taux = deepBatons(10)
#res = courbeBaton(20)
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
