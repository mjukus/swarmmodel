# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:46:26 2020

@author: thesq
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import matplotlib.animation as animation
import scipy.constants as const

'''----------GLOBAL PARAMETERS----------
N is number of particles, tFinal is the simulation time, timestep is the timestep,
partMass is the particle mass, epsilon and sigma are Lennard-Jones parameters, and fixingFactor
ensures that the pairwise distance between a particle and itself doesn't break the code.

Code flags to change aspects of the output are _3D (all particles at z=0 if False), plot3D
(True for 3D plotting), limits (autoscales plots if False) and plotFrames (how many plots desired in
one every x form, Nt for only starting and final positions).
'''
N = 100
tFinal = 50
timestep = 0.01
Nt = int(np.ceil(tFinal/timestep)) #number of timesteps
partMass = 39.948 * const.atomic_mass * np.ones((N,1))
epsilon = 120 * const.Boltzmann
sigma = 0.34E-9
fixingFactor = 0.01E-9

_3D = True
plot3D = True
limits = True
plotFrames = Nt


def plots(x,y,c=None,xLabel="x / m",yLabel="y / m",zLabel="z / m"):
    '''
    Parameters
    ----------
    x : Array of x-values to plot.
    y : Array of y-values to plot.
    c : Array of z-values for plotting or colour mapping, optional.
        The default is None.
    xLabel : Label for the x-axis. The default is "x / m".
    yLabel, zLabel: Analogous to xLabel for y- and z-axes.

    Returns
    -------
    None.

    '''
    #generic plotting function
    limit = 1E-6
    plotFig = plt.figure()
    if plot3D:
        plotAx = plotFig.add_subplot(111,projection="3d")
        plotAx.scatter(x,y,c)
        plotAx.set_zlabel(zLabel)
        if limits:
            plotAx.set_zlim([-limit,limit])
    else:
        plotAx = plotFig.add_subplot(111)
        plotAx.scatter(x,y,c=c)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if limits:
        plotAx.set_xlim([-limit,limit])
        plotAx.set_ylim([-limit,limit])
   
def initialise():
    #starting positions for all particles
    meanPos = 0
    standDev = 3E-6
    randPos = standDev * np.random.randn(N,3) + meanPos
    
    xStart = randPos[:,0:1]
    yStart = randPos[:,1:2]
    if _3D:
        zStart = randPos[:,2:3]
    else:
        zStart = np.zeros((N,1))
    plots(xStart,yStart,zStart)
    
    v = np.zeros((N,3))
    
    return xStart,yStart,zStart,np.hstack((xStart,yStart,zStart)),v

def LJForce(r):
    #Lennard-Jones force calculation
    A = 4 * epsilon * sigma**12
    B = 4 * epsilon * sigma**6
    
    LJForce =  (12*A * r**-13) - (6*B * r**-7)
    LJForce -= LJForce[0][0] * np.identity(N)
    return LJForce

def acceleration(x,y,z):
    
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    
    #print("--------\n---dz----\n",dz)
    
    r = (dx**2 + dy**2 + dz**2)**0.5
    theta = np.nan_to_num(np.arccos(dz * r**-1))
    #print("\n----Theta--\n",theta,"\n",np.cos(theta))
    phi = np.arctan2(dy,dx)
    r += fixingFactor * np.identity(N) 
    
    force = LJForce(r)
    
    #print("\n----Force---\n",force)
    
    invMass = 1 / partMass
    
    a_x = np.sin(theta) * np.cos(phi) * force @ invMass
    a_y = np.sin(theta) * np.sin(phi) * force @ invMass
    a_z = np.cos(theta) * force @ invMass
    
    return np.hstack((a_x,a_y,a_z))

def main():
    
    t = 0
    
    x,y,z,pos,v = initialise()
    
    a = acceleration(x,y,z)
    
    
    for i in range(Nt):
        #print("\n-----Velocity-----\n",v)
        v += a * timestep / 2.0
        #print(v)
        #print("\n----Position----\n",pos)
        pos += v * timestep
        #print("\n",pos)
        #print("\n----Acceleration---\n",a)
        a = acceleration(pos[:,0:1],pos[:,1:2],pos[:,2:3])
        v += a * timestep / 2.0
        t += timestep
        if i % (plotFrames - 1) == 0:
            plots(pos[:,0:1],pos[:,1:2],pos[:,2:3])

main()