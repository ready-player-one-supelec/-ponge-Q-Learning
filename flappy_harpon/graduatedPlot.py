#! /usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns


abscisses = list(range(0, 7001, 500))

def f(path) :
    medians = [-20]
    low = [0]
    up = [0]

    for i in abscisses[1:] :
        with open(path + "testing{}with5000.out".format(i), "r") as f :
            L = f.readlines()
        liste = []
        for i in range(len(L)) :
            toto = L[i][:-1].split(" ")
            if len(toto) == 2 :
                try :
                    vic = int(toto[0])
                    defeat = int(toto[1])
                except :
                    pass
                else :
                    liste.append(vic - defeat)

        liste.sort()
        # confidence intervals
        confidence = 75
        alpha = 100 - confidence
        median = np.median(liste)
        lower = median - np.percentile(liste, alpha / 2)
        upper = np.percentile(liste,100 - alpha / 2) - median
        medians.append(median)
        low.append(lower)
        up.append(upper)
    return medians, low, up

decalage = 10
M1,L1,U1 = f("Saved_Networks/WithoutMaxPooling/Graduated7000/")
abscisses1 = [i - decalage for i in abscisses]
M2,L2,U2 = f("Saved_Networks/WithMaxPooling/Graduated7000/")
abscisses2 = [i + decalage for i in abscisses]

fig = plt.figure()
sns.set()
plt.title("Relative score in accordance with the number of training games")
plt.xlabel("Number of training games")
plt.ylabel("Relative score (scored - conceded)")
plt.ylim(-22,22)
plt.xlim(0,7100)
plt.plot(abscisses1, M1, "b-", label="Without Max Pooling")
plt.errorbar(abscisses1, M1, [L1,U1], color="b", linestyle="None")
plt.plot(abscisses2, M2, "r-", label="With Max Pooling")
plt.errorbar(abscisses2 , M2, [L2,U2], color="r", linestyle="None")
plt.legend()

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
fig.set_size_inches(20, 10)
plt.savefig("Saved_Networks/WithMaxPooling/Graduated7000/comparisonWith_Without.png", dpi = 300)
plt.show()
