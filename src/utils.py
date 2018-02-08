# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 21:35:13 2018

@author: yboetz
"""

import numpy as np

# Calculates centre of momentum
def centreOfMomentum(vel, masses):
    com = np.einsum('ij,i', vel[:,:3], masses)
    M = np.einsum('i->', masses)
    return com / M

# Calculates centre of mass
def centreOfMass(pos):
    masses = pos[:,3]
    com = np.einsum('ij,i', pos[:,:3], masses)
    M = np.einsum('i->', masses)
    return com / M
