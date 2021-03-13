# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 04:12:57 2021
@author: mawga
I have a habit of overcommenting. Forgive me. Thanks. Also, PEP 8...not followed - Jack
"""

import numpy as np # importing numpy

import initialise # importing bespoke functions from local directory
import interactions
import constraints
import tools

from numba import jit # importing tools for improving runtime and timing
from time import perf_counter


def main(axisN: int, nRod: int, partAxisSep: float, Nt: int, timestep: float,
         rodLength: float=2E-6, partMass: float=1E-15, lennardJonesFlag: bool=True,
         epsilon: float=4E-21, sigma: float=1E-6, forceCap: float=5E-15,
         hydrodynamics: bool=True, swimmingSpeed: float=20.4E-6, hydrodynamicThrust: float=0.57E-12,
         viscosity: float=1E-3):
    '''
    Prepares a system of N particles before calculating accelerations and velocities and iterating over Nt timesteps. The positions are stored in an array and are saved to file after the final timestep. Throughout "particle" refers to a rod made up of nRod "interaction points".
    Parameters
    ----------
    axisN : int
        The number of particles on each axis of a cube of particles. The total number of particles, N, is axisN ** 3.
    nRod : int
        The number of interaction points in each particle in metres. Must be greater than 1.
    partAxisSep : float
        The initial distance between two adjacent particles on an axis.
    Nt : int
        The number of timesteps over which the swarm should be simulated.
    timestep : float
        The size of the timestep in seconds.
    rodLength : float, optional
        The length of each particle, head to tail, in metres. The default is 2 μm.
    partMass : float, optional
        The mass of each particle in kg. The default is 1E-15 kg.
    lennardJonesFlag : bool, optional
        Whether or not to calculate and use a Lennard-Jones potential. The default is True.
    epsilon : float, optional
        The value of the Lennard-Jones well depth, ε, in J. The default is 4E-21 J.
    sigma : float, optional
        The value of the Lennard-Jones diameter, σ, in metres. The default is 1 μm.
    forceCap : float, optional
        Caps the repulsive force from the Lennard-Jones potential. It is defined in Newtons and takes a value of 5 fN by default.
    hydrodynamics : bool, optional
        Whether or not to calculate and use the hydrodynamic approximation. The default is True.
    swimmingSpeed : float, optional
        The speed of each particle in metres per second. The default is 20.4 μm/s.
    hydrodynamicThrust : float, optional
        The hydrodynamic thrust from each particle's movement in Newtons. The default is 0.57 pN.
    viscosity : float, optional
        The viscosity of the fluid in the system in Pa S. The default is 1E-3.
    '''
    N = axisN ** 3 # the total number of particles in the system.
    bondLength = rodLength / (nRod-1) # bond length between adjacent points in rod. nRod-1 because there is a point at 0. 
    pointMass = partMass/nRod #the mass of each point in a particle
    invPointMass = 1 / pointMass #inverse mass of each point in a particle
    t = 0 # sets the time to zero at the start
    hydrodynamicThrust = hydrodynamicThrust / nRod # divides thrust by nRod to give the thrust from one interaction point
    #bondStiffness = 1 # for angle constraints. Likely unnecessary.
    #cutoff = 2 * sigma # truncation point above which Lennard-Jones potential is assumed zero; unimplemented and likely unnecessary
    
    '''
    INITIALISATION
    --------------
    Creates the system by producing a grid of particles using the parameters above, and calculates initial velocities. Also creates a file for storage of the output.
    '''
    initStart = perf_counter() # start timing the initialisation of the system
    pos, bondDir = initialise.init(axisN,partAxisSep,nRod,bondLength)
    
    '''
    INTERACTIONS
    ------
    Each timestep, the forces acting on each point in every particle are calculated and act on the points to
    change the system. The "forces" are additive and are a Lennard-Jones potential between particles, a hydrodynamic approximation, particle self-propulsion and an infinite potential well. TO BE IMPLEMENTED: INFINITE POTENTIAL WELL
    '''
    
    def acceleration(pos,r,sepDir): 
        
        if lennardJonesFlag == True:
            LJForce = interactions.lennardJones(r,epsilon,sigma,forceCap) # calls Lennard-Jones function
            
            a_x = invPointMass * sepDir[0] * - LJForce
            a_y = invPointMass * sepDir[1] * - LJForce
            a_z = invPointMass * sepDir[2] * - LJForce
            
            a_x = np.sum(a_x.reshape(N*nRod,-1),axis=1).reshape(N,nRod)
            a_y = np.sum(a_y.reshape(N*nRod,-1),axis=1).reshape(N,nRod)
            a_z = np.sum(a_z.reshape(N*nRod,-1),axis=1).reshape(N,nRod)
            
            a = np.array([a_x,a_y,a_z])
            
        else:
            a = np.zeros((3,N,nRod))
            
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
        
        if hydrodynamics == True:
            hydroVelocity = interactions.hydrodynamic_velocity(viscosity,hydrodynamicThrust,bondDir,r,sepDir)
            velocity = swimmingVelocity + hydroVelocity
        else:
            velocity = swimmingVelocity
            
        return velocity
    
    r, sepDir = tools.separation(pos,N,nRod) # calculates separations between all points and directions of these separations
    baseVelocity = velocity(pos,r,sepDir)
    vAccel = np.zeros((N,nRod,3))
    a = acceleration(pos,r,sepDir)
    
    data = np.zeros((Nt+1,3,N,nRod)) # array describing the positions of all points over time
    data[0] = np.array([pos[:,:,0],pos[:,:,1],pos[:,:,2]]) # adds the initial positions to data
    
    dirData = np.zeros((Nt+1,3,N)) # array describing the positions of all points over time
    bondDir = np.moveaxis(bondDir,1,0)
    dirData[0] = np.array([bondDir[0],bondDir[1],bondDir[2]])
    
    initEnd = perf_counter()
    runtime = initEnd - initStart
    print(f"\nSystem initialised in {runtime:.3f} seconds.\n")
    
    mainStart = perf_counter()
    
    for i in range(Nt):
                
        print(f"Calculating timestep: {i+1} of {Nt}...", end="\r")
        vAccel += a * timestep / 2.0
        pos += (vAccel + baseVelocity) * timestep
        pos, bondDir = constraints.bondCon(pos,bondLength,nRod) # sharply constrains the bonds to bondLength
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
        bondDir = np.moveaxis(bondDir,1,0)
        dirData[i+1] = np.array([bondDir[0],bondDir[1],bondDir[2]])
    
    mainEnd = perf_counter()
    runtime = mainEnd - mainStart
    print(f"\n\nCalculations completed for {N} particles ({N * nRod} interaction points) over {t:.1E} seconds with a timestep of {timestep:.1E} seconds ({Nt} timesteps).\nRun time: {runtime:.3f} seconds.")
    
    print("\nSaving to file...")
    np.save("positions",data)
    np.save("directions",dirData)
    print("Complete.")

#main(3,4,3E-6,200,1E-5)