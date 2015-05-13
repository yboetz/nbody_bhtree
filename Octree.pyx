import numpy as np
cimport numpy as np
from libcpp cimport bool
from time import time

cdef extern from "Octree.h":
    cdef cppclass Octree:
        Octree(float*, float*, int, float, float) except +
        void integrate(float)
        void integrateNSteps(float, int)
        float energy()
        float angularMomentum()
   

cdef class OTree:
    cdef Octree *thisptr
    
    def __cinit__(self, np.ndarray[np.float32_t, ndim=1, mode="c"] pos, np.ndarray[np.float32_t, ndim=1, mode="c"] vel, int n, np.float32_t th, np.float32_t eps2):
        self.thisptr = new Octree (&pos[0], &vel[0], n, th, eps2)
        
    def __dealloc__(self):
        del self.thisptr
    
    def integrate(self, np.float32_t dt):
        self.thisptr.integrate(dt)
    
    def integrateNSteps(self, np.float32_t dt, int n):
        self.thisptr.integrateNSteps(dt, n)
    
    def energy(self):
        return self.thisptr.energy()
    
    def angularMomentum(self):
        return self.thisptr.angularMomentum()