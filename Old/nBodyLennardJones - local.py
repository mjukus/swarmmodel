# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:46:26 2020

@author: thesq
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

N = 100
tFinal = 10000
timestep = 0.1
partMass = 39.948 * const.atomic_mass * np.ones((N,1))
epsilon = 120 * const.Boltzmann
sigma = 0.34E-9
fixingFactor = 0.01E-9

def plots(x,y,c=None,xLabel="x / m",yLabel="y / m"):
    #generic plotting function
    plotFig = plt.figure()
    plotAx = plotFig.add_subplot(111)
    plotAx.scatter(x,y,c=c/max(c))
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    axes = plt.gca()
    limit = 5E-5
    axes.set_xlim([-limit,limit])
    axes.set_ylim([-limit,limit])
   
def initialise():
    #starting positions for all particles
    meanPos = 0
    standDev = 3E-6
    randPos = standDev * np.random.randn(N,3) + meanPos
    
    xStart = randPos[:,0:1]
    yStart = randPos[:,1:2]
    zStart = randPos[:,2:3]
    plots(xStart,yStart,zStart)
    
    v = np.zeros((N,3))
    
    return xStart,yStart,zStart,randPos,v

def LJForce(r):
    #Lennard-Jones force calculation
    A = 4 * epsilon * sigma**12
    B = 4 * epsilon * sigma**6
    
    LJForce = (A * r**-13) - (B * r**-7)
    LJForce -= LJForce[0][0] * np.identity(N)
    return LJForce

def acceleration(x,y,z):
    
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    
    r = (dx**2 + dy**2 + dz**2)**0.5
    r += fixingFactor * np.identity(N)
    
    force = LJForce(r)
    
    invMass = 1 / partMass
    
    a_x = force * dx * r**-1 @ invMass
    a_y = force * dy * r**-1 @ invMass
    a_z = force * dz * r**-1 @ invMass
    
    return np.hstack((a_x,a_y,a_z))

def main():
    
    t = 0
    
    x,y,z,pos,v = initialise()
    
    a = acceleration(x,y,z)
    
    Nt = int(np.ceil(tFinal/timestep))
    
    for i in range(Nt):
        v += a * timestep / 2.0
        pos += v * timestep
        a = acceleration(pos[:,0:1],pos[:,1:2],pos[:,2:3])
        v += a * timestep / 2.0
        t += timestep
        if i % 10 == 0:
            plots(pos[:,0:1],pos[:,1:2],pos[:,2:3])

main()
