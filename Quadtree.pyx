import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef extern from "Quadtree.h":
    cdef cppclass Quadtree:
        Quadtree(float*, int, float*, float) except +
        void integrate(float*, float*, float, float)
        void integrateNSteps(float*, float*, float, float, int)
        void traverse()
   

cdef class QTree:
    cdef Quadtree *thisptr
    
    def __cinit__(self, np.ndarray[np.float32_t, ndim=1, mode="c"] pos, int n, np.ndarray[np.float32_t, ndim=1, mode="c"] cent, np.float32_t th):
        self.thisptr = new Quadtree (&pos[0], n, &cent[0], th)
        
    def __dealloc__(self):
        del self.thisptr
    
    def integrate(self, np.ndarray[np.float32_t, ndim=1, mode="c"] pos, np.ndarray[np.float32_t, ndim=1, mode="c"] vel, np.float32_t dt, np.float32_t eps2):
        self.thisptr.integrate(&pos[0], &vel[0], dt, eps2)
    
    def integrateNSteps(self, np.ndarray[np.float32_t, ndim=1, mode="c"] pos, np.ndarray[np.float32_t, ndim=1, mode="c"] vel, np.float32_t dt, np.float32_t eps2, int n):
        self.thisptr.integrateNSteps(&pos[0], &vel[0], dt, eps2, n)
    
    
    def traverse(self):
        self.thisptr.traverse()
        