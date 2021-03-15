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
         rodLength: float=2E-6, partMass: float=1E-15, swimmingSpeed: float=20.4E-6, tumbleFreq: int=100,
         lennardJonesFlag: bool=True, epsilon: float=4E-21, sigma: float=1E-6, forceCap: float=5E-15,
         hydrodynamics: bool=True, hydrodynamicThrust: float=0.57E-12, viscosity: float=1E-3):
    '''
    Prepares a system of N particles before calculating accelerations and velocities and iterating over Nt timesteps. The positions are stored in an array and are saved to file after the final timestep. Throughout "particle" refers to a rod made up of nRod "interaction points".
     
    System Parameters
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
    
    Particle Parameters
    ----------
    rodLength : float, optional
        The length of each particle, head to tail, in metres. The default is 2 μm.
    partMass : float, optional
        The mass of each particle in kg. The default is 1E-15 kg.
    swimmingSpeed : float, optional
        The speed of each particle in metres per second. The default is 20.4 μm/s.
    tumbleFreq : int, optional
        The number of timesteps between tumbles. The default is 100.
    
    Interaction Parameters
    ----------
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
    INTERACTIONS
    ------------
    Interactions are handled in the following functions. Acceleration is calculated from a Lennard-Jones
    potential and an potential well. Velocity is calculated from a base velocity (in the function velocity())
    made up of a self-propulsive term and a hydrodynamic approximation and the velocity from the acceleration.
    TO BE IMPLEMENTED: potential well
    '''
    def acceleration(pos,r,sepDir): 
        '''
        Calculates the acceleration on each point in the system from a Lennard-Jones potential and a potential well, using the positions of the points and separations between them.

        Parameters
        ----------
        pos : N x nRod x 3 array
            The positions of all the points in the system.
        r : N x nRod x nRod x (N - 1) array
            The distances between all the points and other points not in the same particle.
        sepDir : N x nRod x nRod x (N - 1) array
            The directions of the separations between all points and other points not in the same particle.

        Returns
        -------
        a : N x nRod x 3 array
            The accelerations of all the points in the system.
        '''
        
        if lennardJonesFlag == True:
            # calls the Lennard-Jones calculation if turned on
            LJForce = interactions.lennardJones(r,epsilon,sigma,forceCap)
            
            a_x = invPointMass * sepDir[0] * - LJForce # calculates acceleration on each axis from each force
            a_y = invPointMass * sepDir[1] * - LJForce
            a_z = invPointMass * sepDir[2] * - LJForce
            
            a_x = np.sum(a_x.reshape(N*nRod,-1),axis=1).reshape(N,nRod) # sums together forces on the same point
            a_y = np.sum(a_y.reshape(N*nRod,-1),axis=1).reshape(N,nRod)
            a_z = np.sum(a_z.reshape(N*nRod,-1),axis=1).reshape(N,nRod)
            
            a = np.array([a_x,a_y,a_z])
            
        else:
            a = np.zeros((3,N,nRod)) # no acceleration
            
        a = np.transpose(a,[1,2,0])
        
        return a
    
    
    def velocity(pos,r,sepDir):
        '''
        Calculates base velocity on each point in the system from self-propulsion and a hydrodynamic approximation, using the positions of the points and separations between them.

        Parameters
        ----------
        pos : N x nRod x 3 array
            The positions of all points in the system.
        r : N x nRod x nRod x (N - 1) array
            The distances between all the points and other points not in the same particle.
        sepDir : N x nRod x nRod x (N - 1) array
            The directions of the separations between all points and other points not in the same particle.

        Returns
        -------
        velocity : N x nRod x 3 array
            The velocities of all the points in the system.
        '''
        
        centre = tools.findCentre(pos)[1]
        centreMag = np.linalg.norm(centre, axis=1)
        # the magnitude of the vectors describing the middle of the particles
        
        bondDir = (centreMag[:,np.newaxis]**-1 * centre) # bond directions as unit vectors
        
        swimmingVelocity = swimmingSpeed * np.repeat([bondDir],nRod,axis=1).reshape(N,nRod,3) # particles swim straight ahead at swiming speed
        
        if hydrodynamics == True:
            # calls the hydrodynamics function if turned on
            hydroVelocity = interactions.hydrodynamic_velocity(viscosity,hydrodynamicThrust,bondDir,r,sepDir)
            velocity = swimmingVelocity + hydroVelocity
        
        else:
            velocity = swimmingVelocity
            
        return velocity
    
    '''
    INITIALISATION
    --------------
    Creates the system by producing a grid of particles using the parameters above, and calculates initial
    velocities. Also creates a file for storage of the output.
    '''
    initStart = perf_counter() # start timing the initialisation of the system
    pos, bondDir = initialise.init(axisN,partAxisSep,nRod,bondLength)
    
    
    
    r, sepDir = tools.separation(pos,N,nRod) # calculates initial separations between all points and directions of these separations
    baseVelocity = velocity(pos,r,sepDir) # initial base velocity
    vAccel = np.zeros((N,nRod,3)) # array describing velocity from acceleration
    a = acceleration(pos,r,sepDir) # initial acceleration
    
    data = np.zeros((Nt+1,3,N,nRod)) # array describing the positions of all points over time
    data[0] = np.array([pos[:,:,0],pos[:,:,1],pos[:,:,2]]) # adds the initial positions to data
    
    dirData = np.zeros((Nt+1,3,N)) # array describing the directions of all points over time
    bondDir = np.moveaxis(bondDir,1,0)
    dirData[0] = np.array([bondDir[0],bondDir[1],bondDir[2]]) # populated with directions
    
    initEnd = perf_counter() # end timing the initialisation of the system
    initRuntime = initEnd - initStart
    print(f"\nSystem initialised in {initRuntime:.3f} seconds.\n")
    
    mainStart = perf_counter() # start timing the timestep calculations
    
    for i in range(Nt):
        # over all timesteps, uses leapfrog method to calculate motion of particles
        print(f"Calculating timestep: {i+1} of {Nt}...", end="\r")
        vAccel += a * timestep / 2.0 # calculates velocity from acceleration
        pos += (vAccel + baseVelocity) * timestep # calculates position from total velocity for the current timestep
        pos, bondDir = constraints.bondCon(pos,bondLength,nRod) # sharply constrains the bonds to bondLength
        if (i + 1) % tumbleFreq == 0:
            pos = tools.tumble(pos,bondLength)
        r,sepDir = tools.separation(pos,N,nRod) # new separations
        baseVelocity = velocity(pos,r,sepDir) # new base velocity
        a = acceleration(pos,r,sepDir) # new acceleration
        vAccel += a * timestep / 2 # new velocity from acceleration
        t += timestep # increases time by timestep
        data[i+1] = np.array([pos[:,:,0],pos[:,:,1],pos[:,:,2]]) #adds the positions for the current timestep to data
        bondDir = np.moveaxis(bondDir,1,0)
        dirData[i+1] = np.array([bondDir[0],bondDir[1],bondDir[2]]) # adds the directions of the particles for the current timestep to dirData
    
    mainEnd = perf_counter() # end timing the timestep calculations
    runtime = mainEnd - mainStart
    print(f"\n\nCalculations completed for {N} particles ({N * nRod} interaction points) over {t:.1E} seconds with a timestep of {timestep:.1E} seconds ({Nt} timesteps).\nCalculation time: {runtime:.3f} seconds.\n\nTotal runtime: {(initRuntime + runtime):.3f} seconds.") # output key parameters and timing
    
    print("\nSaving to file...")
    np.save("positions",data) # saves positions and directions as .npy files, for use in analysis.py
    np.save("directions",dirData)
    print("Complete.")

#main(3,4,3E-6,2000,1E-6) # uncomment for running in spyder