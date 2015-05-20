# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:38:23 2015

@author: somebody
"""

import numpy as np
from Octree import OTree
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from time import time
#import os
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
from numba import jit, void, i4, f4

# Function to roll the array self.lineColors by 1.
@jit(void(f4[:,:,:], i4, i4), nopython = True)
def rollLineColor(a, x, y):
    """Only use to roll the array self.lineColors. This function is faster than np.roll, 
    but can only be used in this special case. Will segfault if the shape is not
    given correctly."""
    for i in range(x):
        for j in range(y-1,0,-1):
            a[i,j,0] = a[i,j-1,0]
            a[i,j,1] = a[i,j-1,1]
            a[i,j,2] = a[i,j-1,2]
            a[i,j,3] = a[i,j-1,3]

# Generic thread which takes a function and its arguments and runs it
class WorkerThread(QtCore.QThread):
    def __init__(self, function, *args, **kwargs):
        QtCore.QThread.__init__(self)
        self.function = function
        self.args = args
        self.kwargs = kwargs
 
    def __del__(self):
        self.wait()
 
    def run(self):
        self.function(*self.args,**self.kwargs)
        return

# GL widget class to display nbody data
class NBodyWidget(gl.GLViewWidget): 
    # Create fps signal. Has to be declared outside of __init__
    fpsSignal = QtCore.pyqtSignal(str) 
    
    def __init__(self):
        super(NBodyWidget, self).__init__()
        self.init()
        
    def init(self):
        # Timestep
        self.dt = np.float32(.005)
        # Softening length squared
        self.e = np.float32(.05**2)
        # Tickrate. 1000/max framerate
        self.tickRate = 1000./60
        # Number of intermediate update steps before updating
        self.burst = 1
        # Set distance to origin
        self.opts['distance'] = 20
        # Opening angle
        self.theta = np.float32(.5)
            
        # Create GridItems
        self.gx = gl.GLGridItem()
        self.gx.rotate(90, 0, 1, 0)
        self.gx.translate(-10, 0, 0)
        self.gy = gl.GLGridItem()
        self.gy.rotate(90, 1, 0, 0)
        self.gy.translate(0, -10, 0)
        self.gz = gl.GLGridItem()
        self.gz.translate(0, 0, -10)
                
        # Initial read in of positions and velocity from file
        self.read('Data/Plummer_4096')
        # Initialize Octree
        self.oct = OTree(self.pos, self.vel, self.n, self.theta, self.e)
        
        # Create variable for GLLinePlotItem
        self.lp = None
        
        # Set sizes according to mass, scaled wrt highest mass
        self.size = 75
        self.sizeArray = self.size / 1000 * (self.pos[3::4] / np.amax(self.pos[3::4]))**(1/3)
        # Set colors
        self.colors = [1,1,.5,1]
        self.lineColors = [1,1,.5,1]
        # Set initial line length
        self.lineLength = 50
                
        # Add scatterplot with position data. Needs 3 vectors, self.pos is 4 aligned
        self.sp = gl.GLScatterPlotItem(pos=self.pos.reshape((self.n,4))[:,0:3], size = self.sizeArray,
                                       color = self.colors, pxMode=False)
        self.addItem(self.sp)
        
        # Timer which calls update function at const framerate
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateScatterPlot)
               
        # Init fps counter
        self.fps = 1000 / self.tickRate
        self.timer.timeout.connect(self.fpsCounter)
                                              
    # Calls integrate fucntion and updates data
    def updateScatterPlot(self):
        self.oct.integrateNSteps(self.dt, self.burst)
        self.sp.setData(pos=self.pos.reshape((self.n,4))[:,0:3], 
                        size = self.sizeArray, color = self.colors)
        
    # Initial setup of GLLinePlotItem
    def setupLinePlot(self):
        self.lineData = np.zeros((self.n, self.lineLength, 3), dtype = np.float32)
        self.lineData[:,:,:] = self.pos.reshape((self.n,4))[:,None,0:3]       
        # Add line plot
        self.lp = gl.GLLinePlotItem(pos = self.lineData, color = self.lineColors, 
                                    antialias = True, mode = 'lines', width = 1.5)
        self.addItem(self.lp)
        if np.array(self.colors).size > 4:
            self.toggleLineColors()
    
    # Updates GLLinePlotItem data
    def updateLinePlot(self):
        self.lineData = np.roll(self.lineData, 1, axis = 1)
        self.lineData[:,0,:] = self.pos.reshape((self.n,4))[:,0:3]
        self.lp.setData(pos = self.lineData, color = self.lineColors, antialias = True)
        
    # Deletes GLLinePlotItem for new data read in
    def delLinePlot(self):
        if self.lp in self.items:
            self.timer.timeout.disconnect(self.updateLinePlot)
            self.removeItem(self.lp)
            self.lp = None
            del self.lineData
        else:
            self.lp = None
            try:
                del self.lineData
            except Exception:
                pass
    
    # Sets length of lines
    def setLineLength(self, length):
        try:
            length *= 2
            if self.lp in self.items:
                if length <= self.lineLength:
                    self.lineData = self.lineData[:,:length,:]
                    if np.array(self.lineColors).size > 4:
                        self.lineColors = self.lineColors[:,:length,:]
                else:
                    self.lineData = np.pad(self.lineData, ((0,0),(0,length-self.lineLength),(0,0)), 
                                           'edge')
                    if np.array(self.lineColors).size > 4:
                        self.lineColors = np.pad(self.lineColors, ((0,0),(0,length-self.lineLength),(0,0)), 
                                                 'edge')
            self.lineLength = length
        except Exception as error:
            print(error)
            pass
        
    # Starts/stops timer
    def toggleTimer(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.lastTime = time()
            self.timer.start(self.tickRate)
    
    # Adds/removes GLLinePlotItem
    def toggleLinePlot(self):
        if self.lp in self.items:
            self.timer.timeout.disconnect(self.updateLinePlot)
            self.removeItem(self.lp)
            if np.array(self.lineColors).size > 4:
                self.timer.timeout.disconnect(self.updateLineColors)
                self.lineColors = [1,1,.5,1]              
        else:
            self.setupLinePlot()   
            self.timer.timeout.connect(self.updateLinePlot)
    
    # Toggle grid
    def toggleGrid(self):
        grids = [self.gx, self.gy, self.gz]
        for grid in grids:
            if grid in self.items:
                self.removeItem(grid)
            else:
                self.addItem(grid)
    
    # Toggle colors
    def toggleColors(self):
        if np.array(self.colors).size > 4:
            self.timer.timeout.disconnect(self.updateColors)
            self.colors = [1,1,.5,1]
            self.sp.setGLOptions('additive')
            self.sp.setData(pos=self.pos.reshape((self.n,4))[:,0:3], 
                            size = self.sizeArray, color = self.colors)
            if np.array(self.lineColors).size > 4:
                self.toggleLineColors()
        else:
            self.colors = np.ones((self.n, 4))
            self.updateColors()
            self.timer.timeout.connect(self.updateColors)
            self.sp.setGLOptions('translucent')
            self.sp.setData(pos=self.pos.reshape((self.n,4))[:,0:3], 
                            size = self.sizeArray, color = self.colors)
            if self.lp in self.items:
                self.toggleLineColors()

    # Toggle colors
    def toggleLineColors(self):
        if np.array(self.lineColors).size > 4:
            self.timer.timeout.disconnect(self.updateLineColors)
            self.lineColors = [1,1,.5,1]
            self.lp.setData(pos = self.lineData, color = self.lineColors, 
                            antialias = True)
        elif self.lp in self.items and np.array(self.colors).size > 4:
            self.lineColors = np.ones((self.n, self.lineLength, 4), dtype = np.float32)
            self.lineColors[:,:,2] = .5
            self.updateLineColors()
            self.timer.timeout.connect(self.updateLineColors)
            self.lp.setData(pos = self.lineData, color = self.lineColors, 
                            antialias = True)
    
    # Update dot color depending of current direction of travel
    def updateColors(self):
        vel = self.vel.reshape((self.n,4))[:,0:3]
        tmp = np.sqrt(np.sum(vel * vel, axis = 1))
        self.colors[:,0:3] = (vel[:,:] / tmp[:,None] + 1.1) / 2.1
        self.colors[:,3] = 1
        
    # Update line colors with color at current position
    def updateLineColors(self):
#        self.lineColors = np.roll(self.lineColors, 1, axis = 1)
        rollLineColor(self.lineColors, self.n, self.lineLength)
        self.lineColors[:,0,:] = self.colors[:,:]
    
    # Resets colors
    def resetColors(self):
        if np.array(self.colors).size > 4:
            self.timer.timeout.disconnect(self.updateColors)
            self.colors = [1,1,.5,1]
            self.sp.setGLOptions('additive')
        if np.array(self.lineColors).size > 4:
            self.timer.timeout.disconnect(self.updateLineColors)
            self.lineColors = [1,1,.5,1]

    # Set dot size
    def setSize(self, size):
        self.size = size
        self.sizeArray = (size / 1000) * (self.pos[3::4] / np.amax(self.pos[3::4]))**(1/3)
        self.sp.setData(pos=self.pos.reshape((self.n,4))[:,0:3], 
                        size = self.sizeArray, color = self.colors)
    
    # Set dt
    def setdt(self, dt):
        self.dt = np.float32(dt / 10000)
    
    # Set burst
    def setBurst(self, burst):
        self.burst = burst
            
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
        
    # Function to call when reading in new file.
    def readFile(self, path):
        if self.timer.isActive():
            self.timer.stop()
        try:
            self.read(path)
            del self.oct
            self.oct = OTree(self.pos, self.vel, self.n, self.theta, self.e)
            self.delLinePlot()
            self.resetColors()
            self.setSize(self.size)
            self.resetCenter()
        except FileNotFoundError as error:
            print(error)
        except Exception:
            print('Read error. Data should be aligned as \'m x y z vx vy vz\'.')
    
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
              
    # Calculates current fps
    def fpsCounter(self):
        self.now = time()
        dt = self.now - self.lastTime
        self.lastTime = self.now
        s = np.clip(dt*2., 0, 1)
        self.fps = self.fps * (1-s) + (1.0/dt) * s
        self.fpsSignal.emit('Fps: %.2f' %(self.fps))
    
    # Resets camera center to (0,0,0)
    def resetCenter(self):
        center = -self.opts['center']  
        self.pan(center.x(), center.y(), center.z())
    
    # Stores mouse position
    def mousePressEvent(self, ev):
        super(NBodyWidget, self).mousePressEvent(ev)

    # Resets pan & zoom positon
    def mouseReleaseEvent(self, ev):
        super(NBodyWidget, self).mouseReleaseEvent(ev)
        self.prevZoomPos = None
        self.prevPanPos = None

    # Pans in xy-plane
    def mouseMoveEvent(self, ev):
        """ Allow Shift to Move and Ctrl to Pan."""
        shift = ev.modifiers() & QtCore.Qt.ShiftModifier
        ctrl = ev.modifiers() & QtCore.Qt.ControlModifier
        if shift:
            y = ev.pos().y()
            if not hasattr(self, 'prevZoomPos') or not self.prevZoomPos:
                self.prevZoomPos = y
                return
            dy = y - self.prevZoomPos
            def delta():
                return -dy * 5
            ev.delta = delta
            self.prevZoomPos = y
            self.wheelEvent(ev)
        elif ctrl:
            pos = ev.pos().x(), ev.pos().y()
            if not hasattr(self, 'prevPanPos') or not self.prevPanPos:
                self.prevPanPos = pos
                return
            dx = pos[0] - self.prevPanPos[0]
            dy = pos[1] - self.prevPanPos[1]
            self.pan(dx, dy, 0, relative=True)
            self.prevPanPos = pos
        else:
            super(NBodyWidget, self).mouseMoveEvent(ev)
    
    # Makes a test, calculates energy and momentum drift after some steps
    def testFunction(self, dt, num):              
        print('Testing: ', end="")
        E0, J0 = self.oct.energy(), self.oct.angularMomentum()

        T = time()
        self.oct.integrateNSteps(np.float32(dt), num)
        T = time() - T

        E1, J1 = self.oct.energy(), self.oct.angularMomentum()
        dE, dJ = E1 - E0, J1 - J0
        print('Did %i cycles in %.3f seconds.' %(num, T))
        print('E0 =  %.3f, E1 = %.3f\nJ0 = %.3f, J1 = %.3f' %(E0,E1,J0,J1))
        print('dE = %.3f = %.3f E0\ndJ = %.3f = %.3f J0\n'
              %(dE, dE / E0, dJ, dJ / J0))        
    
    # Calls testFunction in separate thread
    def test(self, dt, num):
        self.worker = WorkerThread(self.testFunction, dt, num)
        self.worker.start()
    

class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.init()
     
    def init(self):   
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        
        # Initialize N-body GLWidget, buttons and sliders
        self.GLWidget = NBodyWidget()
        self.GLWidget.keyPressEvent = self.keyPressEvent
        # Buttons
        startButton = QtGui.QPushButton('Start/stop (s)')
        startButton.clicked.connect(self.GLWidget.toggleTimer)
        closeButton = QtGui.QPushButton('Close (Esc)')
        closeButton.clicked.connect(pg.exit)
        lineButton = QtGui.QPushButton('Draw lines (l)')
        lineButton.clicked.connect(self.GLWidget.toggleLinePlot)
        gridButton = QtGui.QPushButton('Toggle grid (g)')
        gridButton.clicked.connect(self.GLWidget.toggleGrid)
        # Slider to change size
        sizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        sizeSlider.setMinimum(1)
        sizeSlider.setMaximum(200)
        sizeSlider.setValue(int(self.GLWidget.size))
        sizeSlider.valueChanged.connect(self.GLWidget.setSize)
        sizeSliderLabel = QtGui.QLabel('Change size', self)
        # Slider to change dt
        dtSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        dtSlider.setMinimum(1)
        dtSlider.setMaximum(500)
        dtSlider.setValue(int(self.GLWidget.dt*10000))
        dtSlider.valueChanged.connect(self.GLWidget.setdt)
        dtSliderLabel = QtGui.QLabel('Change dt', self)
        # Slider to change number of bursts
        burstSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        burstSlider.setMinimum(1)
        burstSlider.setMaximum(100)
        burstSlider.setValue(int(self.GLWidget.burst))
        burstSlider.valueChanged.connect(self.GLWidget.setBurst)
        burstSliderLabel = QtGui.QLabel('Change burst', self)   
        # Slider to change line length
        lengthSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        lengthSlider.setMinimum(1)
        lengthSlider.setMaximum(100)
        lengthSlider.setValue(self.GLWidget.lineLength // 2)
        lengthSlider.valueChanged.connect(self.GLWidget.setLineLength)
        lengthSliderLabel = QtGui.QLabel('Change line length', self)  
        # Display fps
        fpsLabel = QtGui.QLabel('Fps: %.2f' %(self.GLWidget.fps), self)      
        self.GLWidget.fpsSignal.connect(fpsLabel.setText)
        # Add widgets on grid
        grid.addWidget(self.GLWidget, 1, 3, 50,50)    
        grid.addWidget(startButton, 1, 2)
        grid.addWidget(closeButton, 1, 1)
        grid.addWidget(lineButton, 2, 2)
        grid.addWidget(gridButton, 2, 1)
        grid.addWidget(sizeSlider, 11, 1, 1, 2)
        grid.addWidget(sizeSliderLabel, 10, 1, 1, 2)
        grid.addWidget(dtSlider, 8, 1, 1, 2)
        grid.addWidget(dtSliderLabel, 7, 1, 1, 2)
        grid.addWidget(burstSlider, 5, 1, 1, 2)
        grid.addWidget(burstSliderLabel, 4, 1, 1, 2)
        grid.addWidget(lengthSlider, 14, 1, 1, 2)
        grid.addWidget(lengthSliderLabel, 13, 1, 1, 2)
        grid.addWidget(fpsLabel, 16, 1, 1, 1)
    

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.init()
        self.show()
        
    def init(self):
        self.resize(1536, 960)
        self.setWindowTitle('N-body simulation')
        
        self.window = Window()
        self.window.keyPressEvent = self.keyPressEvent
        self.setCentralWidget(self.window)
#        self.statusBar()
        # Menubar entries
        openFile = QtGui.QAction(QtGui.QIcon('open.png'), 'Open file', self)
        openFile.setShortcut('O')
        openFile.setStatusTip('Read data from file')
        openFile.triggered.connect(self.showDialog)  
        closeApp = QtGui.QAction(QtGui.QIcon('quit.png'), 'Quit', self)
        closeApp.setShortcut('Escape')
        closeApp.setStatusTip('Exits application')
        closeApp.triggered.connect(pg.exit)
        # Add menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)  
        fileMenu.addAction(closeApp)
        
    # Show file dialog and calls file read function
    def showDialog(self):
        if self.window.GLWidget.timer.isActive():
            self.window.GLWidget.timer.stop()
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open file','Data/')
        self.window.GLWidget.readFile(path)
        
    # Define keypress events
    def keyPressEvent(self, e):
        if e.isAutoRepeat():
            pass
        elif e.key() == QtCore.Qt.Key_C:
            print(self.window.GLWidget.centreOfMass())
            print(self.window.GLWidget.centreOfMomentum())
        # Start and stop timer with S
        elif e.key() == QtCore.Qt.Key_S:
            self.window.GLWidget.toggleTimer()
        elif e.key() == QtCore.Qt.Key_L:
            self.window.GLWidget.toggleLinePlot()
        elif e.key() == QtCore.Qt.Key_G:
            self.window.GLWidget.toggleGrid()
        elif e.key() == QtCore.Qt.Key_N:
            self.window.GLWidget.toggleColors()
        elif e.key() == QtCore.Qt.Key_R:
            self.window.GLWidget.resetCenter()
        elif e.key() == QtCore.Qt.Key_E:
            print('E = %.3f, J = %.3f' %(self.window.GLWidget.oct.energy(), 
                                         self.window.GLWidget.oct.angularMomentum()))
        elif e.key() == QtCore.Qt.Key_T:
            if self.window.GLWidget.timer.isActive():
                self.window.GLWidget.timer.stop()
            
            while True:
                text, ok = QtGui.QInputDialog.getText(self, 'Testing', 'Enter number of cycles')
                if text.isdigit() and ok:
                    break
                elif not ok:
                    break
            
            if ok:
                num = int(text)
                self.window.GLWidget.test(0.01, num)


if __name__ == "__main__":
    # Start Qt applicatoin
    app = QtGui.QApplication([])
    # Create main window
    win = MainWindow()
    app.exec()
    pg.exit()