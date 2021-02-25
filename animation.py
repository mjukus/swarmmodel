# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:48:09 2020

@author: thesq
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tools

def animate(frames,data,plot):
    
    for i in range(data[0,1].shape[0]):
        plot[i]._offsets3d = (data[frames,0:1,i],data[frames,1:2,i],data[frames,2:3,i])
    #plot._offsets3d = (data[frames,0:1],data[frames,1:2],data[frames,2:3])
    
    return plot

def main(data):

    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    
    scatter = [ ax.scatter(data[0,0:1,i],data[0,1:2,i],data[0,2:3,i]) for i in range(data[0,1].shape[0]) ]
    
    frames = len(data)
    
    ani = animation.FuncAnimation(fig, animate, frames, interval=20, fargs=(data,scatter))
    
    g