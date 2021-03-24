# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 02:22:10 2021

@author: thesq
"""

import numpy as np
from numba import jit
import tools

@jit(nopython=True) # just in time compiles using numba in C to give a speed increase
def lennardJones(r,epsilon: float,sigma: float,forceCap: float):
    '''
    Calculates the force from a Lennard-Jones potential on any number of particles using their separations. Positive (repulsive) forces can be capped.

    Parameters
    ----------
    r : array
        The separations from which the force will be calculated. r in the Lennard-Jones equation.
    epsilon : float
        The Lennard-Jones well depth.
    sigma : float
        The Lennard-Jones effective diameter.
    forceCap : float
        The largest allowed positive value of a calculated force, to which all greater values will be capped.

    Returns
    -------
    force : array
        An array of forces of the same shape as r.

    '''
    
    rMinus6 = r**-6 # precalculates the 6th powers. Think it makes it quicker; depends how smart numpy is.
    #cutoffMinus6 = cutoff**-6
    sigmaPower6 = sigma**6
    
    A = 4 * epsilon * sigmaPower6**2 # repulsion prefactor in potential
    B = 4 * epsilon * sigmaPower6 # attraction prefactor in potential
    
    #potential = A*rMinus6**2 - B*rMinus6
    #shift = cutoff**-1 * (12*A*cutoffMinus6**2 - 6*B*cutoffMinus6)
    
    force = r**-1 * (12*A*rMinus6**2 - 6*B*rMinus6) #- shift

    force = force.flatten() # numba doesn't support boolean slicing of multi-dimensional arrays, amongst many other things
    force[force > forceCap] = forceCap # so it flattens the array, slices and caps
    force = force.reshape(r.shape) # then shapes as before.
    
    return force

#r = np.linspace(1E-7,5E-6,5000)
#force = lennardJones(r,4E-21,1E-6,5E-15) # lines for plotting Lennard-Jones force equation
#tools.plot(r,force)

def well(pos, wellCoef):
    
    force = - wellCoef * pos
    
    return force

#@jit(forceobj=True) # seems to slow the function down
def hydrodynamic_velocity(viscosity,force,forceDirection,separation,separationDirection):
    # separationDirection is an array of the directions between points in a particle to points in other particles, for each particle
    # forceDirection is the direction the rod is swimming in
    shape = separationDirection.shape[1:]
    directionalDependence = np.empty(shape)
    for i in range(shape[0]):
        # iterates over particles, doing a dot product between particle and separation directions. There are of course more separation directions (N * nRod * nRod per particle) than particle directions (1 per particle)
        directionalDependence[i] = np.einsum("i,i...->...",forceDirection[i],separationDirection[:,i])
        # gives cosines of the angles between directions because inputs are unit vectors
    
    hydroVelocity = ((force/(np.pi * 8 * viscosity * (separation**2))) * ((3*(directionalDependence**2))-1)) * separationDirection #calculation
    
    shape = hydroVelocity.shape
    hydroVelocity = np.sum(hydroVelocity.reshape((shape[0]*shape[1]*shape[2],-1)),axis=1).reshape((3,-1))
    hydroVelocity = hydroVelocity.T.reshape(shape[1],shape[2],3)
    # ugly code to sum together contributions to the velocity on each point
    
    return hydroVelocity
