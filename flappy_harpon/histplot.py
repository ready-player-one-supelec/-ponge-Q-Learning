#! /usr/bin/python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def hist(filename) :
    with open(filename, "r") as f :
        L = f.readlines()
    del L[:6]
    del L[-1]
    #del L[::2]
    liste = []
    for i in range(len(L)) :
        toto = L[i][:-2].split(" ")
        if len(toto) == 2 :
            try :
                vic = int(toto[0])
                defeat = int(toto[1])
            except :
                pass
            else :
                liste.append(vic - defeat)
    plt.title("Number of games = {} ; max = {}".format(len(liste),max(liste)))
    plt.xlim(-25,25)
    bins = np.arange(-21, 23) - 0.5
    ax = plt.gca()
    sns.distplot(liste, bins = bins, hist = True, kde = False, norm_hist = False, ax = ax)
    ax2 = plt.twinx()
    sns.distplot(liste, bins=bins, hist=False, kde=True, norm_hist=True, kde_kws={"bw":1, "color":"r"}, ax = ax2)
    ax2.grid(None)
    ax.set_xlabel("Relative score (goals scored - goals conceded)")
    ax.set_ylabel("Number of games")
    ax2.set_ylabel("Density")
    ax.yaxis.label.set_color('b')
    ax.tick_params(axis='y', colors='b')
    ax2.yaxis.label.set_color('r')
    ax2.tick_params(axis='y', colors='r')

fig = plt.figure()
sns.set()
path = "Saved_Networks/WithoutMaxPooling/15000/"
hist(path+"testing15000with5000.out")
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
fig.set_size_inches(20, 10)
plt.savefig(path+"hist_kde.png", dpi = 300)
plt.show()
