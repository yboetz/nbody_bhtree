# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:22:10 2015

@author: somebody
"""

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from Octree import OTree
from time import time

# Main class to display window
class Window(gl.GLViewWidget):  
    def __init__(self):
        super(Window, self).__init__()
        self.init()
        self.show()
        
    def init(self):
        self.dt = np.float32(.01)
        self.eps2 = np.float32(0.01**2)
        self.center = np.array([0,0], dtype = np.float32)
        self.theta = np.float32(.25)
        self.frate = 60
        
        
        # Set distance to origin
        self.opts['distance'] = 20
        # Set title
        self.setWindowTitle('N-body simulation')
#        self.n = pos.size // 4
                   
        # Show 2D grid in xy-plane
        grid = gl.GLGridItem()
        self.addItem(grid)
                
        # Initialize position and velocity vectors and fill with random data
        self.read('Data/GalaxyMerger_4096')
        self.size = .1

        self.oct = OTree(self.pos, self.n, self.center, self.theta)
        # Add scatterplot with data
        self.sp = gl.GLScatterPlotItem(pos=self.pos.reshape((self.n, 4))[:,:3], size = self.size,
                                       color = [1,1,1,1], pxMode=False)
        self.addItem(self.sp)
        
        # Start timer which calls update function at const framerate
        self.t = QtCore.QTimer()
        self.t.timeout.connect(self.updateData)
           
    # Integrates one step forward and sets data new    
    def updateData(self):
#        st = time()
        self.oct.integrateNSteps(self.pos, self.vel, self.dt, self.eps2, 1)
#        print(time() - st)
        self.sp.setData(pos=self.pos.reshape((self.n, 4))[:,:3], size = self.size, color = [1,1,1,1])
    
    # Calculates centre of momentum
    def centreOfMomentum(self):
        masses = self.pos[3::4]
        tmp = np.sum(self.vel.reshape((self.n,4))[:,0:3] * masses[:, None], axis = 0)
        M = np.sum(masses)
        return tmp / M
    
    # Calculates centre of mass
    def centreOfMass(self):
        masses = self.pos[3::4]
        tmp = np.sum(self.pos.reshape((self.n,4))[:,0:3] * masses[:, None], axis = 0)
        M = np.sum(masses)
        return tmp / M
    
     # Reads in data from file
    def read(self, path):
        with open(path, 'r') as file:
            for i, l in enumerate(file):
                pass
            n = i + 1
            file.seek(0)
            pos = np.zeros((n, 4), dtype = np.float32)
            vel = np.zeros((n, 4), dtype = np.float32)
            for i, line in enumerate(file):
                x = list(map(float, line.split()))
                pos[i,0:3] = x[1:4]        
                pos[i,3] = x[0]
                vel[i,0:3] = x[4:7]
            vel[:,3] = 0
            self.pos = pos.reshape(4*n)
            self.vel = vel.reshape(4*n)
            self.n = n
            # Go to centre of mass & momentum system
            self.pos -= np.tile(np.append(self.centreOfMass(),0),self.n)
            self.vel -= np.tile(np.append(self.centreOfMomentum(),0),self.n)
                   
    def keyPressEvent(self, e):
        if e.isAutoRepeat():
            pass
        elif e.key() == QtCore.Qt.Key_Escape:
            self.close()
        elif e.key() == QtCore.Qt.Key_S:
            if self.t.isActive():
                self.t.stop()
            else:
                self.t.start(1000/self.frate)

if __name__ == "__main__":
    # Start Qt applicatoin
    app = QtGui.QApplication([])
    # Create window
    win = Window()
    app.exec()
    pg.exit()
     

       