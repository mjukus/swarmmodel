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

from numba import jit
from time import perf_counter
#import animation

'''
KEY PARAMETERS
--------------
These parameters define the system. TO BE IMPLEMENTED: INPUTS? AND REALISTIC NUMBERS.
'''
axisN = 6 # the number of particles on each axis of a cube. Used to create a grid of particles at the start.
N = axisN ** 3
partAxisSep = 1.5E-5 # the axial separation of each particle on the cube from the next.

rodLength = 2E-6 # length of each rod-like particle, approx 2µm. Diameter approx 1µm, Vol approx 1µm**3
nRod = 4 # number of interaction points in rod. Must be greater than 1
bondLength = rodLength / (nRod-1) # bond length between adjacent points in rod. nRod-1 because there is a point at 0. 
bondStiffness = 1

partMass = 1E-15 #the mass of each whole rod-like particle, approx 1 picogram
pointMass = partMass/nRod #the mass of each point in a particle
invPointMass = 1 / pointMass #inverse mass of each point in a particle

Nt = 200 # number of timesteps
timestep = 1E-6 # size of timestep, in seconds
t = 0 # sets the time to zero at the start
plotFrames = 10

epsilon = 4E-25 # the Lennard-Jones parameters
sigma = 1E-6
cutoff = 2 * sigma # truncation point above which potential is assumed zero
forceCap = 2E-21

swimmingSpeed = 20.4E-6 # The hydrodynamics parameters, Speed should be approx 20.4 µm/s
hydrodynamicThrust = 0.57E-12 / nRod #Should be approx 0.57 pN
viscosity = 1


'''
INITIALISATION
--------------
Creates the system by producing a grid of particles using the parameters above. Also creates a file for storage
of the output.
'''
initStart = perf_counter()
pos = initialise.init(axisN,partAxisSep,nRod,bondLength)

'''
INTERACTIONS
------
Each timestep, the forces acting on each point in every particle are calculated and act on the points to
change the system. The "forces" are additive and are a Lennard-Jones potential between particles, a hydrodynamic
approximation, particle self-propulsion and an infinite potential well. TO BE IMPLEMENTED: ALL OF IT.
'''

def acceleration(pos,r,sepDir): 
    
    LJForce = interactions.lennardJones(r,epsilon,sigma,forceCap) # calls Lennard-Jones function
    
    a_x = invPointMass * sepDir[0] * (- LJForce)
    a_y = invPointMass * sepDir[1] * (- LJForce)
    a_z = invPointMass * sepDir[2] * (- LJForce)
    
    a_x = np.sum(a_x.reshape(N*nRod,-1),axis=1).reshape(N,nRod)
    a_y = np.sum(a_y.reshape(N*nRod,-1),axis=1).reshape(N,nRod)
    a_z = np.sum(a_z.reshape(N*nRod,-1),axis=1).reshape(N,nRod)
    
    #a_x, a_y, a_z = (np.einsum("ijkl->ij", a_x), np.einsum("ijkl->ij", a_y), np.einsum("ijkl->ij", a_z))
    a = np.array([a_x,a_y,a_z])
    a = np.transpose(a,[1,2,0])
    
    return a


def velocity(pos,r,sepDir):
    
    particleTails = pos[:,0] # the positions of the two ends of the particles - the heads and tails
    particleHeads = pos[:,-1]
    centre = 0.5 * (particleHeads - particleTails)
    # the mean of the head and tail position relative to the tail - the centre of the particle
    centreMag = np.linalg.norm(centre, axis=1)
    # the magnitude of the vectors describing the middle of the particles
    
    bondDir = (centreMag[:,np.newaxis]**-1 * centre)
    
    swimmingVelocity = swimmingSpeed * np.repeat([bondDir],nRod,axis=1).reshape(N,nRod,3)
    
    hydroVelocity = interactions.hydrodynamic_velocity(viscosity,hydrodynamicThrust,bondDir,r,sepDir)
    velocity = swimmingVelocity + hydroVelocity
    
    return velocity

r, sepDir = tools.separation(pos,N,nRod)
baseVelocity = velocity(pos,r,sepDir)
vAccel = np.zeros((N,nRod,3))
a = acceleration(pos,r,sepDir)

data = np.zeros((Nt+2,3,N,nRod)) # array describing the positions of all points over time
data[0] = np.array([pos[:,:,0],pos[:,:,1],pos[:,:,2]]) # adds the initial positions to data

initEnd = perf_counter()
runtime = initEnd - initStart
print(f"\nSystem initialised in {runtime:.3f} seconds.\n")

mainStart = perf_counter()

for i in range(Nt+1):
    
    print(f"Calculating timestep: {i} of {Nt}...", end="\r")
    vAccel += a * timestep / 2.0
    pos += (vAccel + baseVelocity) * timestep
    pos = constraints.bondCon(pos,bondLength,nRod) # sharply constrains the bonds to bondLength
    r,sepDir = tools.separation(pos,N,nRod)
    baseVelocity = velocity(pos,r,sepDir)
    a = acceleration(pos,r,sepDir)
    vAccel += a * timestep / 2.0
    t += timestep
    

    '''
    CONSTRAINTS
    -----------
    After the forces have acted on all the particles and points for each timestep, the constraint functions
    constrain first the angles of the particles to keep the rods straight, followed by the lengths of the
    particles. TO BE IMPLEMENTED: ANGLE CONSTRAINTS.
    '''
    
    data[i+1] = np.array([pos[:,:,0],pos[:,:,1],pos[:,:,2]]) #adds the positions for the current timestep to data

mainEnd = perf_counter()
runtime = mainEnd - mainStart
print(f"\n\nCalculations completed for {N * nRod} interaction points over {t:.1E} seconds with a timestep of {timestep:.1E} seconds.\nRun time: {runtime:.3f} seconds.")

print("\nSaving to file...")
np.save("output",data)
print("Complete.")
#animation.main(data.reshape(Nt+2,3,N*nRod)) # calls the animation function. It is janky.