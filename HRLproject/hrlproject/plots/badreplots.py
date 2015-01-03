"""Plots for Badre data."""

import os

import numpy as np
import pylab as plt

def hier_activation():
    flat = None
    flat_activations = None
    for i in range(0,5):
        with open(os.path.join("..","..","data","badre","flat_spiking","dataoutput_%s.txt" % i)) as f:
            data = [[[float(v) for v in entry.split(" ")] for entry in r.split(";")] for r in f.readlines()]
        flat_trange = np.asarray([x[0] for x in data[0]]) 
        activation = np.asarray([(x[1]+y[1])/2 for x,y in zip(data[6],data[7])])
        flat_activations = activation if flat_activations is None else np.vstack((flat_activations,activation))
        flat = activation if flat is None else activation+flat
    flat /= flat_activations.shape[0]

    hierarchy = None
    hierarchy_activations = None
    for i in range(0,5):
        with open(os.path.join("..","..","data","badre","hierarchical_spiking","dataoutput_%s.txt" % i)) as f:
            data = [[[float(v) for v in entry.split(" ")] for entry in r.split(";")] for r in f.readlines()]
        hierarchy_trange = np.asarray([x[0] for x in data[0]]) 
        activation = np.asarray([(x[1]+y[1])/2 for x,y in zip(data[6],data[7])])
        hierarchy_activations = activation if hierarchy_activations is None else np.vstack((hierarchy_activations,activation))
        hierarchy = activation if hierarchy is None else activation+hierarchy
    hierarchy /= hierarchy_activations.shape[0]
    
    fig = plt.figure(figsize=[6,4])
    ax = plt.axes([0.13,0.13,0.75,0.75])
    plt.plot(flat_trange, flat, "-", color='red', linewidth=3)
    plt.plot(hierarchy_trange, hierarchy, "-", color='green', linewidth=3)

    plt.xlabel("time (s)")
    plt.ylabel("activation")
    plt.legend(["flat", "hierarchical"], borderpad=0.3,
               loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_yaxis().tick_left()
    ax.get_xaxis().tick_bottom()
#    plt.xlim([0,5500])
#    plt.ylim([0,50])
    
#    plt.savefig(r"D:\Documents\officesvn\writing\cogsci2014\figures\hier_vs_flat.png")
#    plt.savefig(r"D:\Documents\officesvn\writing\cogsci2014\figures\hier_vs_flat.pdf")

    plt.figure(figsize=[6,4])
    plt.plot(flat_trange, flat_activations.T)
    
    plt.figure(figsize=[6,4])
    plt.plot(hierarchy_trange, hierarchy_activations.T)
    
def reward_acc():
    flat = None
    flat_rewards = None
    flat_trange = None
    for i in range(0,5):
        with open(os.path.join("..","..","data","badre","flat_spiking","dataoutput_%s.txt" % i)) as f:
            data = [[[float(v) for v in entry.split(" ")] for entry in r.split(";")] for r in f.readlines()]
        flat_trange = np.asarray([x[0] for x in data[0]])
        reward = np.cumsum([x[1] for x in data[0]])
        flat_rewards = reward if flat_rewards is None else np.vstack((flat_rewards,reward))
        flat = reward if flat is None else reward+flat
    flat /= flat_rewards.shape[0]

    hierarchy = None
    hierarchy_rewards = None
    hierarchy_trange = None
    for i in range(0,5):
        with open(os.path.join("..","..","data","badre","hierarchical_spiking","dataoutput_%s.txt" % i)) as f:
            data = [[[float(v) for v in entry.split(" ")] for entry in r.split(";")] for r in f.readlines()]
        hierarchy_trange = np.asarray([x[0] for x in data[0]])
        reward = np.cumsum([x[1] for x in data[0]])
        hierarchy_rewards = reward if hierarchy_rewards is None else np.vstack((hierarchy_rewards,reward))
        hierarchy = reward if hierarchy is None else reward+hierarchy
    hierarchy /= hierarchy_rewards.shape[0]
    
    print np.max(np.diff(flat))
    print np.max(np.diff(hierarchy))
    print np.max(np.diff(np.diff(flat)))
    print np.max(np.diff(np.diff(hierarchy)))
    
    plt.figure()
    plt.plot(flat_trange, flat)
    plt.plot(hierarchy_trange, hierarchy)
    
    plt.figure()
    plt.plot(flat_trange, flat_rewards.T)
    
    plt.figure()
    plt.plot(hierarchy_trange, hierarchy_rewards.T)

hier_activation()
#reward_acc()

plt.show()