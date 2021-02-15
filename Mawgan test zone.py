# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:53:45 2020

@author: mawga
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:46:26 2020

@author: thesq
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

N = 10
tFinal = 20
timestep = 0.01
partMass = 39.948 * const.atomic_mass * np.ones((N,1))
epsilon = 120 * const.Boltzmann
sigma = 0.34E-9
fixingFactor = 0.001E-9

def plots(x,y,xLabel="x / m",yLabel="y / m"):
    #generic plotting function
    plotFig = plt.figure()
    plotAx = plotFig.add_subplot(111)
    plotAx.scatter(x,y)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    axes = plt.gca()
    limit = 1E-6
    axes.set_xlim([-limit,limit])
    axes.set_ylim([-limit,limit])
   
def initialise():
    #starting positions for all particles
    meanPos = 0
    standDev = 5E-7
    randPos = standDev * np.random.randn(N,3) + meanPos
    
    xStart = randPos[:,0:1]
    yStart = randPos[:,1:2]
    zStart = randPos[:,2:3]
    plots(xStart,yStart)
    
    v = np.zeros((N,3))
    
    return xStart,yStart,zStart,randPos,v

def LJForce(dx,dy,dz):
    #Lennard-Jones force calculation
    A = 4 * epsilon * sigma**12
    B = 4 * epsilon * sigma**6
    
    LJForce_x = (A * dx**-13) - (B * dx**-7)  
    LJForce_x -= LJForce_x[0][0] * np.identity(N)
    LJForce_y = (A * dy**-13) - (B * dy**-7)
    LJForce_y -= LJForce_y[0][0] * np.identity(N)
    LJForce_z = (A * dz**-13) - (B * dz**-7)
    LJForce_z -= LJForce_z[0][0] * np.identity(N)
    
    return LJForce_x,LJForce_y,LJForce_z

def acceleration(x,y,z):
    fixing = np.full((10,10),0.01E-9)
    dx = x.T - x + fixing
    dy = y.T - y + fixing
    dz = z.T - z + fixing
    
    forces = LJForce(dx,dy,dz)
   
    invMass = 1 / partMass
    
    a_x = forces[0] @ invMass
    a_y = forces[1] @ invMass
    a_z = forces[2] @ invMass
    

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
            plots(pos[:,0:1],pos[:,1:2])

main()