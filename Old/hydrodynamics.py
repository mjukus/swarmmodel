# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:39:12 2020

@author: thesq
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
_3D = False
   
def initialise():
    #starting positions for all particles
    meanPos = 0
    standDev = 1E-6
    randPos = standDev * np.random.randn(N,3) + meanPos
    
    xStart = randPos[:,0:1]
    yStart = randPos[:,1:2]
    if _3D:
        zStart = randPos[:,2:3]
    else:
        zStart = np.zeros((N,1))
    
    
    v = np.zeros((N,3))
    
    return xStart,yStart,zStart,np.hstack((xStart,yStart,zStart)),v

def hydrodynamic_velocity(viscosity,Force,ForceDirection,Seperation,SeperationDirection):
    
    P = Force  # im stupid
    DirectionalDependence = np.dot(ForceDirection,SeperationDirection)  # Essentially angle between force and position 
    #print(DirectionalDependence)
    velocity = ((P/(np.pi * 8 * viscosity * (Seperation**2))) * ((3*(DirectionalDependence**2))-1)) * SeperationDirection #calculation
    #print (SeperationDirection)
    #print(velocity)
    
    return velocity

fluidDensity = 1000
lengthScale = 1E-6
typicalSpeed = 30E-6
viscosity = 0.001
velocity = 0
N = 100000
reynolds = fluidDensity * lengthScale * typicalSpeed / viscosity
Force = 5E-12               # Force magnitude of force dipole
velocity = np.zeros([N,2])  # Initialises 2D Array of velocities
pos_1 = np.array([0,0])     # Origin
ForceDirection = np.array([1,0]) # Direction of force dipole

x,y,z,pos,v = initialise()  # Gives random positions for slicin
pos_2 = pos[0:N,0:2]

for i in range(N):

    SeperationPos = pos_2[i] - pos_1 # Distance between pos 1 and 2
    Seperation = np.sqrt(SeperationPos.dot(SeperationPos))  # Magnitude of seperation distance
    SeperationDirection = SeperationPos/Seperation  # unit vector of seperation
    #print(hydrodynamic_velocity(viscosity,Force,ForceDirection,Seperation,SeperationDirection))
    vel = hydrodynamic_velocity(viscosity,Force,ForceDirection,Seperation,SeperationDirection)  #calculates velocity and truncates it
    if np.sqrt(vel.dot(vel)) > 8000:
        vel = ([0,0])
    else:    
        velocity[i] = vel
    

#print(velocity)
#print(velocity)

limit = 5E-7
print (velocity)
plt.quiver(pos_2[0:N,0],pos_2[0:N,1],velocity[0:N,0],velocity[0:N,1])   # Creates vector field graph and plots
plt.xlim(-limit,limit)
plt.ylim(-limit,limit)
plt.show()
