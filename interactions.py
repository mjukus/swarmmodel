# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 02:22:10 2021

@author: thesq
"""

import numpy as np
from numba import jit

@jit
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
    force[force > forceCap] = forceCap # so I flatten the array, slice and set to zero
    force = force.reshape(r.shape) # then shape as before.
    
    return force

def hydrodynamic_velocity(viscosity,Force,ForceDirection,Seperation,SeperationDirection):
     
    # Force magnitude of force dipole defined by swimming speed
    # Force dipole direction is seperate and simply the direction the rod is swimming in
    #DirectionalDependence = np.dot(ForceDirection,SeperationDirection) 
    DirectionalDependence = np.einsum("ij,ik...->k...",ForceDirection.T,SeperationDirection)
    
    # Essentially angle between force dipole direction and separation direction from point on another rod
    #print(DirectionalDependence)
    hydro_velocity = ((Force/(np.pi * 8 * viscosity * (Seperation**2))) * ((3*(DirectionalDependence**2))-1)) * SeperationDirection #calculation
    hydro_velocity = np.transpose(np.einsum("ijklm->ijk",hydro_velocity),[1,2,0])
    #print (SeperationDirection)
    #print(velocity)
    
    return hydro_velocity
