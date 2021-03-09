# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 02:22:10 2021

@author: thesq
"""

import numpy as np
from numba import jit
import tools

@jit(nopython=True)
def lennardJones(r,epsilon,sigma,forceCap,cutoff=False):
    
    rMinus6 = r**-6
    #cutoffMinus6 = cutoff**-6
    sigmaPower6 = sigma**6
    
    A = 4 * epsilon * sigmaPower6**2
    B = 4 * epsilon * sigmaPower6
    
    #potential = A*rMinus6**2 - B*rMinus6
    #shift = cutoff**-1 * (12*A*cutoffMinus6**2 - 6*B*cutoffMinus6)
    
    force = r**-1 * (12*A*rMinus6**2 - 6*B*rMinus6) #- shift

    force = force.flatten() # numba doesn't support boolean slicing of multi-dimensional arrays, amongst many other things
    force[force > forceCap] = forceCap # so it flattens the array, slices and sets to zero
    force = force.reshape(r.shape) # then shapes as before.
    
    return force

#r = np.linspace(1E-7,5E-6,5000)
#force = lennardJones(r,4E-21,1E-6,5E-15)

#tools.plot(r,force)

#@jit
def hydrodynamic_velocity(viscosity,Force,ForceDirection,Seperation,SeperationDirection):
    # Force magnitude of force dipole defined by swimming speed
    # Force dipole direction is seperate and simply the direction the rod is swimming in
    shape = SeperationDirection.shape[1:]
    DirectionalDependence = np.empty(shape)
    for i in range(shape[0]):
        DirectionalDependence[i] = np.einsum("i,i...->...",ForceDirection[i],SeperationDirection[:,i])
    # Essentially angle between force dipole direction and separation direction from point on another rod
    
    hydro_velocity = ((Force/(np.pi * 8 * viscosity * (Seperation**2))) * ((3*(DirectionalDependence**2))-1)) * SeperationDirection #calculation
    
    shape = hydro_velocity.shape
    hydro_velocity = np.sum(hydro_velocity.reshape((shape[0]*shape[1]*shape[2],-1)),axis=1).reshape((3,-1))
    hydro_velocity = hydro_velocity.T.reshape(shape[1],shape[2],3)
    
    return hydro_velocity
