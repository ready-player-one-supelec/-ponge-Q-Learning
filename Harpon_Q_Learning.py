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
        mouvs = []
        for a in A: #Cette étape peut rallonger fortement le programme il faudra réflechir à l'enlever 
            if a(s) != -1:
                mouvs.append(a)
        aa = choose(s,R,Q,mouvs,opt)
        lA.append(aa)
        s = aa(s)
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
        
def chooseEpsilon(s,R,Q,A,opt): # Pas à jour est pas forcément utile 
    return s
    

#%% Essai simple 
l = 3

def haut(s):
    if s < l :
        return -1
    else:
        return s-l
    
def bas(s):
    if s > (l-1)*l-1 :
        return -1
    else:
        return s+l
    
def gauche(s):
    if s%l == 0 :
        return -1
    else:
        return s-1

def droite(s):
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
    
def deux(s): 
    if s-2 < 1:
        return "d"
    else:
        e = enemi(s)
        res = s-2-e
        if res < 1:
            return "v"
        else:
            return res
        
def un(s): 
    if s-1 < 1:
        return "d"
    else:
        e = enemi(s)
        res = s-1-e
        if res < 1:
            return "v"
        else:
            return res
    
def enemi(s): #Pas a jour !!!
    return rd.randint(1,2)

def enemi_dur(s): #Pas a jour !!!
    return Q[s].index(max(Q[s])) #Il faut try except Q 


def jeuBatons(i): #Pas a jour !!!
    S = ["d",1,2,3,4,5,6,7,8,9,10,11,"v"]
    A = [un,deux]
    R = {"d" :-100,"v":10}
    s0 = 11
    Q = {}
    taux = 0
    for j in range(i):
        [lS,lA] = frontprop(S,A,s0,R,Q,chooseExp)
        Q = backprop(S,A,lS,lA,R,Q,0.5,0.5)
    for j in range(1000):
        [lS,lA] = frontprop(S,A,s0,R,Q,chooseWin)
        taux += lS[-1]/12
        print(lS)
        #print([c(k) for k in lA])
    print(taux/(j+1))
    return Q

