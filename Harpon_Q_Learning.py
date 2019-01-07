# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:18:27 2018

@author: Loic
"""
import random as rd 
import numpy as np

#%% Programme 

# R doit être un dictionnaire indéxé par les états S.
# IMPORTANT On peut remplir seulement les valeurs non nulles
# Q doit être un dictionnaire indéxé par les états S et les actions A. (On peut indexer par des fonctions et c'est cool !)
# IMPOTANT On peut initialiser Q vide et on prendra dans ce cas comme valeur par défaut 0
# ATTENTION Si on run sans recacluler Q cela ne marchera plus car on aura redefini les fonctions de A donc on ne pourra plus acceder à Q
# Il faut donc soit le recalcalculer soit ne pas refefinir les fonctions de A
# On n'a pas en fait besoin de connaitre S ce qui est bien car on ne le connait en général pas 

def frontprop(A,s0,R,Q,choose,opt = 1): 
    s = s0
    lS = [s0]
    lA = []
    try:
        rs = R[s0]
    except KeyError:
        rs = 0 #R n'est pas forcément rempli 
    while rs == 0: 
        mouvs = A
        #mouvs = []
        #for a in A: #Cette étape peut rallonger fortement le programme il faudra réflechir à l'enlever 
        #    if a(R,Q,A,s) != -1:
        #        mouvs.append(a)
        aa = choose(s,R,Q,mouvs,opt)
        lA.append(aa)
        s = aa(R,Q,A,s)
        lS.append(s)
        try:
            rs = R[s]
        except KeyError:
            rs = 0
    return [lS,lA]

def backprop(A,lS,lA,R,Q,la = 0.4,g = 0.3):
    n = len(lS)
    for i in range(n-2,-1,-1):
        s = lS[i]
        ss = lS[i+1]
        a = lA[i]
        try:
            rss = R[ss]
        except KeyError:
            rss = 0
        try:
            qsa = Q[s,a]
        except KeyError:
            qsa = 0
        maxss = 0
        for aa in A:
            try:
                qssaa = Q[ss,aa]
            except KeyError:
                qssaa = 0
            if qssaa > maxss: maxss = qssaa
        Q[s,a] = la*(rss + g*maxss)+(1-la)*qsa
    return Q

def chooseMax(s,R,Q,A,opt):
    res = A[0]
    resval = 0
    for a in A:
        try:
            qsa = Q[s,a]
        except KeyError:
            qsa = 0
        if qsa > resval:
            res =a 
            resval = qsa             
    return res

def chooseBoltz(s,R,Q,A,T):
    summ = 0
    for a in A:
        try:
            qsa = Q[s,a]
        except KeyError:
            qsa = 0
        summ += np.exp(qsa/T)
    rand = rd.random()*summ
    b = 0
    for a in A:
        try:
            qsa = Q[s,a]
        except KeyError:
            qsa = 0
        b += np.exp(qsa/T)
        if rand < b :
            return a
        
def chooseEpsilon(s,R,Q,A,opt): # Pas à jour et pas forcément utile 
    return s 
    
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

def sample(D):
    rd.shuffle(D)
    res = []
    k = 0
    n = len(D)
    while k < n and len(res) < 32:
        if D[k] != []:
            res.append(D[k])
    return res
            
        

def ajoute(D,elt):
    D[rd.randin(0,len(D)-1)] = elt
    return D

def deepQlearning(A,s0,R,choose,memoire,it,neural_it,reseau,Tlim,phi = phibase,gamma = 0.5,rate = 0.001,opt = 0,modify = lambda x: x):
    reseau.append(len(A))
    inputs = s0 #A voir  --> Implementation ATARI pour s0
    (QW,QB) = per.random_w_b(inputs,reseau)
    D = [[] for i in range(memoire)]
    for i in range(it):
        lAS = [s0]
        s = s0
        p = phi(lAS)
        try:
            r = R[s0]
        except KeyError:
            r = 0
        j = 0 #Evite une boucle inifnie mais doit etre élevé pour ne pas bloquer le jeu 
        while j<Tlim and abs(r) < 100 :
            j += 1
            opt = modify(opt) #opt peut etre tout les arguments suplémentaires odnt on a besoin pour le choix 
            a = choose(s,R,(QW,QB),A,opt) #Modify permet de faire evoluer opt par exemple si opt = epsilon on peut le faire décroitre... 
            ss = a(s) #ici on fait l'action --> implementation ATARI
            lAS = lAS + [a,ss]
            pp = phi(lAS)
            try:
                r = R[ss]
            except KeyError:
                r = 0
            D  = ajoute(D,(p,a,r,pp)) #!!
            batch = sample(D)
            inputs = [batch[i][-1] for i in range(len(batch))]
            y = [0 for k in range(len(batch))] # Calcul de Theorical output 
            for k in len(batch): # On doit d'abbord calculer les sorties que l'on connait 
                (sk,ak,rk,ssk) = batch[k]
                if abs(rk) >= 100:
                    y[k] = rk
                else:
                    maxk = max(per.front_prop(batch,reseau,QW,QB,per.tanh))
                    y[k] = rk + gamma*maxk
            front = per.front_prop(inputs,reseau,QW,QB,per.tanh)[-1]
            for k in len(y) :
                front[k][A.index(a)] = y[k]
                y[k] = front
            (QW,QB) = batch_training(inputs,y,reseau,QW,QB,rate,neural_it)
            s = ss
            p = pp
    return (QW,QB)
            


        
        
        
#%% Essai simple 
l = 3

def haut(R,Q,A,s):
    if s < l :
        return -1
    else:
        return s-l
    
def bas(R,Q,A,s):
    if s > (l-1)*l-1 :
        return -1
    else:
        return s+l
    
def gauche(R,Q,A,s):
    if s%l == 0 :
        return -1
    else:
        return s-1

def droite(R,Q,A,s):
    if s%l == l-1 :
        return -1
    else:
        return s+1

def test(i):
    S = [i for i in range(l*l)]
    A = np.array([gauche,droite,haut,bas])
    R = {}    #[0,-10,10,0,-10,0,0,0,0]
    R[2] = 10
    R[1] = -10
    R[4] = -10
    Q = {}
    s0 = 0
    for j in range(i):
        [lS,lA] = frontprop(A,s0,R,Q,chooseBoltz,99)
        Q = backprop(A,lS,lA,R,Q,0.5,0.5)
    return (Q,S)


def affiche(Q):
    print("")
    print(" (0)  "+str(round(Q[0,droite]))+"→←"+"0"+"  (1)  "+"0"+"→←"+"0"+"  (2)")
    print("")
    print(str(round(Q[3,haut]))+"↑↓"+str(round(Q[0,bas]))+"        "+"0"+"↑↓"+"0"+"        "+str(round(Q[5,haut]))+"↑↓"+"0")
    print("")
    print(" (3)  "+str(round(Q[3,droite]))+"→←"+"0"+"  (4)  "+"0"+"→←"+str(round(Q[5,gauche]))+"  (5)")
    print("")
    print(str(round(Q[6,haut]))+"↑↓"+str(round(Q[3,bas]))+"        "+str(round(Q[7,haut]))+"↑↓"+"0"+"        "+str(round(Q[8,haut]))+"↑↓"+str(round(Q[5,bas])))
    print("")
    print(" (6)  "+str(round(Q[6,droite]))+"→←"+str(round(Q[7,gauche]))+"  (7)  "+str(round(Q[7,droite]))+"→←"+str(round(Q[8,gauche]))+"  (8)")
    print("")

#(Q,S) = test(10000)
#affiche(Q)
    
#%% Jeu des batons 
    
def deux(R,Q,A,s): 
    if s-2 < 1:
        return "d"
    else:
        e = ennemi(R,Q,A,s)
        res = s-2-e
        if res < 1:
            return "v"
        else:
            return res
        
def un(R,Q,A,s): 
    if s-1 < 1:
        return "d"
    else:
        e = ennemi(R,Q,A,s)
        res = s-1-e
        if res < 1:
            return "v"
        else:
            return res
    
def ennemi(R,Q,A,s): 
    return rd.randint(1,2)

    
    
def jeuBatons(training): #marche
    S = ["d",1,2,3,4,5,6,7,8,9,10,11,"v"]
    A = [un,deux]
    R = {"d" :-1,"v":1}
    Q = {}
    s0=11
    for j in range(training):
        [lS,lA] = frontprop(A,s0,R,Q,chooseBoltz)
        Q = backprop(A,lS,lA,R,Q)
    return (Q,S)

def afficherBatons(Q):
    for k in range(11,0,-1):
        try:
            print(k, ": ", Q[k,un],Q[k,deux])
        except KeyError:
            print(k, ": ", "etat non atteignable")
    return None


def tauxVictoire(training):
    A = [un,deux]
    R = {"d" :-1,"v":1}
    Q=jeuBatons(training)[0]
    taux=0
    s0=11
    for i in range(1000):
        [lS,lA] = frontprop(A,s0,R,Q,chooseMax)
        if 1 not in lS:
            taux+=1/10
    return taux, "% de reussite"
    
    
#%%
def deux_dur(R,Q,A,s): 
    if s-2 < 1:
        return "d"
    else:
        e = ennemi_dur(R,Q,A,s-2)
        res = s-2-e
        if res < 1:
            return "v"
        else:
            return res
        
def un_dur(R,Q,A,s): 
    if s-1 < 1:
        return "d"
    else:
        e = ennemi_dur(R,Q,A,s-1)
        res = s-1-e
        if res < 1:
            return "v"
        else:
            return res
    
def ennemi_dur(R,Q,A,s): 
    try :
        aa = chooseMax(s,R,Q,A,0)
        if aa==deux_dur:
            return 2
        if aa==un_dur:
            return 1
    except KeyError:
        return rd.randint(1,2)
    


def smartStick(training):
    S = ["d",1,2,3,4,5,6,7,8,9,10,11,"v"]
    A = [un_dur,deux_dur]
    R = {"d" :-2,"v":2}
    Q = {}
    s0=11
    for j in range(training):
        [lS,lA] = frontprop(A,s0,R,Q,chooseBoltz)
        Q = backprop(A,lS,lA,R,Q)
    for j in range(1000):
        [lS,lA] = frontprop(A,s0,R,Q,chooseMax)
    return (Q,S)


def afficherBatonsDur(Q):
    for k in range(11,0,-1):
        try:
            print(k, ": ", Q[k,un_dur],Q[k,deux_dur])
        except KeyError:
            print(k, ": ", "etat non atteignable")
    return None

def tauxVictoireSmart(training):
    A = [un_dur,deux_dur]
    R = {"d" :-1,"v":1}
    Q=smartStick(training)[0]
    taux=0
    s0=11
    for i in range(1000):
        [lS,lA] = frontprop(A,s0,R,Q,chooseMax)
        if 1 not in lS:
            taux+=1/10
    return taux, "% de reussite"
    