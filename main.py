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
axisN = 2 # the number of particles on each axis of a cube. Used to create a grid of particles at the start.
N = axisN ** 3
partAxisSep = 5 # the axial separation of each particle on the cube from the next.

rodLength = 1 # length of each rod-like particle, approx 2µm. Diameter approx 1µm, Vol approx 1µm**3
nRod = 4 # number of interaction points in rod. Must be greater than 1
bondLength = rodLength / (nRod-1) # bond length between adjacent points in rod. nRod-1 because there is a point at 0. 
bondStiffness = 1

partMass = 1 #the mass of each whole rod-like particle, approx 1 picogram
pointMass = partMass/nRod #the mass of each point in a particle
invPointMass = 1 / pointMass #inverse mass of each point in a particle

epsilon = 1 # the Lennard-Jones parameters
sigma = 1
cutoff = 2 * sigma # truncation point above which potential is assumed zero
fixingFactor = 0.01E-15

swimmingSpeed = 1 # The hydrodynamics parameters, Speed should be approx 20.4 µm/s
hydrodynamicThrust = 1 #Should be approx 0.57 pN

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
    dx = x.T - x[:,:,np.newaxis,np.newaxis] #creates 4D! tensors of x, y and z separations
    y = pos[:,:,1:2].reshape((N,nRod))
    dy = y.T - y[:,:,np.newaxis,np.newaxis] #let me take a moment to apologise for my constant reshaping
    z = pos[:,:,2:3].reshape((N,nRod))
    dz = z.T - z[:,:,np.newaxis,np.newaxis] #it cannot possibly be efficient
    
    for i in range(N):
        dx[i,:,:,i] = 0 # ensures that every point in a particle has zero separation
        dy[i,:,:,i] = 0 # from every other in the same particle. Seems kinda pointless
        dz[i,:,:,i] = 0 # now unless all zeros are purged before LJ is calculated.
    
    r = (dx**2 + dy**2 + dz**2)**0.5 # calculate magnitude of separations
    r = r[r != 0].reshape(N,nRod,nRod,N-1) # remove all zeros to avoid nan in force
    
    a = np.zeros((N,nRod,3)) # zero acceleration for now
    
    LJForce = interactions.lennardJones(r,epsilon,sigma) # calls lennard jones function
    LJForce = np.einsum("ijkl->ij",LJForce) # sums all forces on the same point to give resultant LJ force on each
    
    return a


v = 0.01*np.random.randn(N,nRod,3) # random velocities, this is testing code really
#v = np.repeat([v],nRod,axis=1).reshape(N,nRod,3) # same for each point in a particle
a = acceleration(pos)

data = np.zeros((Nt+1,3,N,nRod)) # array describing the positions of all points over time
data[0] = np.array([pos[:,:,0],pos[:,:,1],pos[:,:,2]]) # adds the initial positions to data

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
    
    data[i+1] = np.array([pos[:,:,0],pos[:,:,1],pos[:,:,2]]) #adds the positions for the current timestep to data
    
#animation.main(data.reshape(Nt+1,3,N*nRod)) # calls the animation function. It is janky.
