import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef extern from "../src/octree_mod.cpp":
    cdef cppclass Octree:
        Octree(float*, float*, int, int, float, float) except +
        void integrate(float)
        void integrateNSteps(float, int)
        float energy()
        float angularMomentum()
        void updateColors(float*)
        void updateLineColors(float*, float*, int)
        void updateLineData(float*, int)
        void saveCentreOfMass(float*)
        void saveCentreOfMomentum(float*)
        double T
   

cdef class OTree:
    cdef Octree *thisptr
    
    def __cinit__(self, np.ndarray[np.float32_t, ndim=1, mode="c"] pos, np.ndarray[np.float32_t, ndim=1, mode="c"] vel,
                  int n, int ncrit, np.float32_t th, np.float32_t eps2):
        self.thisptr = new Octree (&pos[0], &vel[0], n, ncrit, th, eps2)
        
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
    
    def updateColors(self, np.ndarray[np.float32_t, ndim=2, mode="c"] col):
        return self.thisptr.updateColors(&col[0,0])
    
    def updateLineColors(self, np.ndarray[np.float32_t, ndim=2, mode="c"] col, np.ndarray[np.float32_t, ndim=3, mode="c"] linecol, int length):
        return self.thisptr.updateLineColors(&col[0,0], &linecol[0,0,0], length)
    
    def updateLineData(self, np.ndarray[np.float32_t, ndim=3, mode="c"] linedata, int length):
        return self.thisptr.updateLineData(&linedata[0,0,0], length)
    
    def centreOfMass(self):
        cdef np.ndarray[dtype = np.float32_t, ndim = 1, mode="c"] com = np.empty(4, dtype = np.float32)
        self.thisptr.saveCentreOfMass(&com[0])
        return com
    
    def centreOfMomentum(self):
        cdef np.ndarray[dtype = np.float32_t, ndim = 1, mode="c"] com = np.empty(4, dtype = np.float32)
        self.thisptr.saveCentreOfMomentum(&com[0])
        return com
    
    property T:
        def __get__(self): return self.thisptr.T