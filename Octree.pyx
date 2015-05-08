import numpy as np
cimport numpy as np
from libcpp cimport bool
from time import time

cdef extern from "Octree.h":
    cdef cppclass Octree:
        Octree(float*, float*, int, float*, float) except +
        void integrate(float, float)
        void integrateNSteps(float, float, int)
        float energy()
        void traverse()
   

cdef class OTree:
    cdef Octree *thisptr
    
    def __cinit__(self, np.ndarray[np.float32_t, ndim=1, mode="c"] pos, np.ndarray[np.float32_t, ndim=1, mode="c"] vel, int n, np.ndarray[np.float32_t, ndim=1, mode="c"] cent, np.float32_t th):
        self.thisptr = new Octree (&pos[0], &vel[0], n, &cent[0], th)
        
    def __dealloc__(self):
        del self.thisptr
    
    def integrate(self, np.float32_t dt, np.float32_t eps2):
        self.thisptr.integrate(dt, eps2)
    
    def integrateNSteps(self, np.float32_t dt, np.float32_t eps2, int n):
        self.thisptr.integrateNSteps(dt, eps2, n)
    
    def energy(self):
        return self.thisptr.energy()
      
    def traverse(self):
        self.thisptr.traverse()