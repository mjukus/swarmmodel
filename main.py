# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:41:56 2021

@author: thesq
"""

import numpy as np
import initialise
import interactions
import constraints
import tools
import animation

'''
KEY PARAMETERS
--------------
These parameters define the system. TO BE IMPLEMENTED: INPUTS? AND REALISTIC NUMBERS.
'''
axisN = 10 # the number of particles on each axis of a cube. Used to create a grid of particles at the start.
N = axisN ** 3
partAxisSep = 5 # the axial separation of each particle on the cube from the next.

rodLength = 1 # length of each rod-like particle, approx 2µm. Diameter approx 1µm, Vol approx 1µm**3
nRod = 4 # number of interaction points in rod. Must be greater than 1
bondLength = rodLength / (nRod-1) # bond length between adjacent points in rod. nRod-1 because there is a point at 0. 
bondStiffness = 1

partMass = 1 #the mass of each whole rod-like particle, approx 1 picogram
pointMass = partMass/nRod #the mass of each point in a particle

epsilon = 1 # the Lennard-Jones parameters
sigma = 1

swimmingSpeed = 1 # The hydrodynamics parameters, Speed should be approx 20.4 µm/s
hydrodynamicThrust = 1 #Should be approx 0.57 pN

Nt = 100 # number of timesteps
timestep = 1 # size of timestep, in seconds
t = 0 # sets the time to zero at the start
plotFrames = 10


'''
INITIALISATION
--------------
Creates the system by producing a grid of particles using the parameters above.
'''
pos = initialise.init(axisN,partAxisSep,nRod,bondLength)


'''
INTERACTIONS
------
Each timestep, the forces acting on each point in every particle are calculated and act on the points to
change the system. The "forces" are additive and are a Lennard-Jones potential between particles, a hydrodynamic
approximation, particle self-propulsion and an infinite potential well. TO BE IMPLEMENTED: ALL OF IT.
'''
def acceleration(pos):
    
    x = pos[:,:,0:1].reshape((N,nRod))
    dx = x.T - x[:,:,np.newaxis,np.newaxis] #creates a 4D! tensor of x separations
    y = pos[:,:,1:2].reshape((N,nRod))
    dy = y.T - y[:,:,np.newaxis,np.newaxis]
    z = pos[:,:,2:3].reshape((N,nRod))
    dz = z.T - z[:,:,np.newaxis,np.newaxis]
    
    for i in range(N):
        dx[i,:,:,i] = 0
        dy[i,:,:,i] = 0
        dz[i,:,:,i] = 0
    
    r = (dx**2 + dy**2 + dz**2)**0.5
    
    a = np.zeros((N,nRod,3)) # zero acceleration for now
    
    LJForce = interactions.lennardJones(r,epsilon,sigma)
    
    return a


v = 0.025*np.random.randn(N,nRod,3) # random velocities, this is testing code really
#v = np.repeat([v],nRod,axis=1).reshape(N,nRod,3) # same for each point in a particle
a = acceleration(pos)

data = np.zeros((Nt+1,3,N,nRod))
data[0] = np.array([pos[:,:,0],pos[:,:,1],pos[:,:,2]])

for i in range(Nt):
    v += a * timestep / 2.0
    pos += v * timestep
    a = acceleration(pos)
    v += a * timestep / 2.0
    t += timestep
    
    
    #if i % (plotFrames - 1) == 0:
    #        tools.plot(np.vstack(pos)[:,0:1],np.vstack(pos)[:,1:2],np.vstack(pos)[:,2:3])

    '''
    CONSTRAINTS
    -----------
    After the forces have acted on all the particles and points for each timestep, the constraint functions
    constrain first the angles of the particles to keep the rods straight, followed by the lengths of the
    particles. TO BE IMPLEMENTED: ANGLE CONSTRAINTS.
    '''
    pos = constraints.bondCon(pos,bondLength,nRod) # sharply constrains the bonds to bondLength
    
    data[i+1] = np.array([pos[:,:,0],pos[:,:,1],pos[:,:,2]])
    
#animation.main(data.reshape(Nt+1,3,N*nRod))
