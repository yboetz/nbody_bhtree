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
    def __init__(self):
        super(NBodyWidget, self).__init__()
        self.init()

    # renderText has to be called inside paintGL
    def paintGL(self, *args, **kwds):
        gl.GLViewWidget.paintGL(self, *args, **kwds)
        self.renderText(30, 30, "Fps:\t%.2f" %(self.fps))
        self.renderText(30, 45, "N:\t%i" %(self.n))
        self.renderText(30, 60, "Step:\t%.4f " %(self.dt))
        self.renderText(30, 75, "Time:\t%.3f" %(self.oct.T))

    def init(self):
        # Timestep
        self.dt = np.float32(.005)
        # Softening length. CANNOT be zero or program will crash
        self.e = np.float32(.05)
        # Opening angle
        self.theta = np.float32(.5)
        # Max number of bodies per critical cell
        self.Ncrit = 64
        # Tickrate. 1000/max framerate
        self.tickRate = 1000./60
        # Number of intermediate update steps before drawing
        self.burst = 1
        # Set distance to origin
        self.opts['distance'] = 35

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
        self.read('Data/Plummer/Plummer_4096')
        # Initialize Octree
        self.oct = OTree(self.pos, self.vel, self.n, self.Ncrit, self.theta, self.e)

        # Set sizes according to mass, scaled wrt highest mass
        self.size = 75
        self.sizeArray = self.size / 1000 * (self.pos[3::4] / np.amax(self.pos[3::4]))**(1/3)
        # Set colors
        self.colors = (1,1,.5,1)
        self.lineColors = (1,1,.5,1)
        self.isColored = False

        # Add scatterplot with position data. Needs 3-vectors, self.pos is 4-aligned
        self.sp = gl.GLScatterPlotItem(pos=self.pos.reshape((self.n,4))[:,0:3], size = self.sizeArray,
                                       color = self.colors, pxMode=False)
        self.addItem(self.sp)
        # Add line plot
        self.lp = gl.GLLinePlotItem()
        # Set initial line length
        self.lineLength = 2

        # Timer which calls update function at const framerate
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateData)
        self.timer.timeout.connect(self.updateScatterPlot)

        # Init fps counter
        self.fps = 1000 / self.tickRate
        self.timer.timeout.connect(self.fpsCounter)

    # Integrates positions & velocities self.burst steps forward
    def updateData(self):
        self.oct.integrateNSteps(self.dt, self.burst)
                                              
    # Updates scatterplot
    def updateScatterPlot(self):
        self.sp.setData(pos=self.pos.reshape((self.n,4))[:,0:3],
                        size = self.sizeArray, color = self.colors)

    # Updates lineplot
    def updateLinePlot(self):
        self.oct.updateLineData(self.lineData, self.lineLength)
        self.lp.setData(pos = self.lineData, color = self.lineColors, antialias = True)
        
    # Setup of GLLinePlotItem
    def setupLinePlot(self):
        self.lineData = np.empty((self.n, self.lineLength, 3), dtype = np.float32)
        self.lineData[:,:,:] = self.pos.reshape((self.n,4))[:,None,0:3]       
        self.lp.setData(pos = self.lineData, color = self.lineColors,
                        antialias = True, mode = 'lines', width = 1.5)
        self.addItem(self.lp)
        self.timer.timeout.connect(self.updateLinePlot)
        if self.isColored:
            self.toggleLineColors()

    # Sets length of lines
    def setLineLength(self, length):
        length *= 2
        if self.lp in self.items:
            if length <= self.lineLength:
                self.lineData = np.array(self.lineData[:,:length,:], dtype=np.float32)
                if self.isColored:
                    self.lineColors = np.array(self.lineColors[:,:length,:], dtype=np.float32)
            else:
                self.lineData = np.array(np.pad(self.lineData, ((0,0),(0,length-self.lineLength),(0,0)),
                                       'edge'), dtype=np.float32)
                if self.isColored:
                    self.lineColors = np.array(np.pad(self.lineColors, ((0,0),(0,length-self.lineLength),(0,0)),
                                             'edge'), dtype=np.float32)
            self.lp.setData(pos = self.lineData, color = self.lineColors, antialias = True)
        self.lineLength = length

    # Starts/stops timer
    def toggleTimer(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.lastTime = time()
            self.timer.start(self.tickRate)
    
    # Toggles scatter/lineplot
    def togglePlot(self):
        if self.lp in self.items:
            self.timer.timeout.disconnect(self.updateLinePlot)
            self.removeItem(self.lp)
            if self.isColored:
                self.timer.timeout.disconnect(self.updateLineColors)
                self.lineColors = (1,1,.5,1)
            self.timer.timeout.connect(self.updateScatterPlot)
            self.addItem(self.sp)
            self.sp.setData(pos=self.pos.reshape((self.n,4))[:,0:3],
                            size = self.sizeArray, color = self.colors)
        else:
            self.timer.timeout.disconnect(self.updateScatterPlot)
            self.removeItem(self.sp)
            self.setupLinePlot()
    
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
        if self.isColored:
            self.timer.timeout.disconnect(self.updateColors)
            self.isColored = False
            self.colors = (1,1,.5,1)
            self.sp.setGLOptions('additive')
        else:
            self.timer.timeout.connect(self.updateColors)
            self.isColored = True
            self.colors = np.empty((self.n, 4), dtype = np.float32)
            self.updateColors()
            self.sp.setGLOptions('translucent')
        if self.sp in self.items:
            self.sp.setData(pos=self.pos.reshape((self.n,4))[:,0:3],
                            size = self.sizeArray, color = self.colors)
        self.toggleLineColors()

    # Toggle colors of lines
    def toggleLineColors(self):
        if self.lp in self.items:
            if self.isColored:
                self.lineColors = np.ones((self.n, self.lineLength, 4), dtype = np.float32)
                self.lineColors[:,:,2] = .5
                self.updateLineColors()
                self.timer.timeout.connect(self.updateLineColors)
            else:
                self.timer.timeout.disconnect(self.updateLineColors)
                self.lineColors = (1,1,.5,1)
            self.lp.setData(pos = self.lineData, color = self.lineColors,
                            antialias = True)

    # Update dot color depending of current direction of travel
    def updateColors(self):
        self.oct.updateColors(self.colors)
        
    # Update line colors with color at current position
    def updateLineColors(self):
        self.oct.updateLineColors(self.colors, self.lineColors, self.lineLength)
    
    # Resets colors
    def resetColors(self):
        if self.isColored:
            self.isColored = False
            self.timer.timeout.disconnect(self.updateColors)
            self.colors = (1,1,.5,1)
            self.sp.setGLOptions('additive')
            if self.lp in self.items:
                self.timer.timeout.disconnect(self.updateLineColors)
                self.lineColors = (1,1,.5,1)

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
        
    # Reads a new file and sets up new octree class. Resets colors and plots
    def readFile(self, path):
        if self.timer.isActive():
            self.timer.stop()
        try:
            self.read(path)
            self.oct = OTree(self.pos, self.vel, self.n, self.Ncrit, self.theta, self.e)
            self.resetColors()
            self.setSize(self.size)
            self.resetCenter()
            self.lineData = None
            if self.lp in self.items:
                self.togglePlot()
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
    
    # Makes a test: Calculates energy and momentum drift after num steps
    def testFunction(self, dt, num):              
        print('Testing: ', end="")
        E0, J0 = self.oct.energy(), self.oct.angularMomentum()

        T = time()
        self.oct.integrateNSteps(np.float32(dt), num)
        T = time() - T

        E1, J1 = self.oct.energy(), self.oct.angularMomentum()
        dE, dJ = E1 - E0, J1 - J0
        print('Did %i cycles in %.3f seconds.' %(num, T))
        print('E0 = %.4f, E1 = %.4f; J0 = %.4f, J1 = %.4f' %(E0,E1,J0,J1))
        print('dE = %.4f = %.4f E0; dJ = %.4f = %.4f J0\n'
              %(dE, dE / E0, dJ, dJ / J0))        
    
    # Calls testFunction in separate thread
    def test(self, dt, num):
        self.worker = WorkerThread(self.testFunction, dt, num)
        self.worker.start()
    
# QWidget class with controls & NBodyWidget
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
        # Labels for controls
        controlLabel = QtGui.QLabel('Controls:\nS\tStart/stop\nE\tPrint energy\n'
                                    'C\tPrint COM\nN\tToggle colors\nL\tToggle dots/lines\n'
                                    'O\tOpen file\nT\tTesting\nEsc\tClose', self)
        # Add widgets on grid
        grid.addWidget(self.GLWidget, 1, 3, 50,50)    
        grid.addWidget(startButton, 1, 2)
        grid.addWidget(closeButton, 1, 1)
        grid.addWidget(sizeSlider, 10, 1, 1, 2)
        grid.addWidget(sizeSliderLabel, 9, 1, 1, 2)
        grid.addWidget(dtSlider, 7, 1, 1, 2)
        grid.addWidget(dtSliderLabel, 6, 1, 1, 2)
        grid.addWidget(burstSlider, 4, 1, 1, 2)
        grid.addWidget(burstSliderLabel, 3, 1, 1, 2)
        grid.addWidget(lengthSlider, 13, 1, 1, 2)
        grid.addWidget(lengthSliderLabel, 12, 1, 1, 2)
        grid.addWidget(controlLabel, 16, 1, 1, 2)
    
# Main window, to have file menu & statusbar
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
            com1 = self.window.GLWidget.oct.centreOfMass()
            com2 = self.window.GLWidget.oct.centreOfMomentum()
            print("R = (%.3f, %.3f, %.3f)" %(com1[0], com1[1], com1[2]), end=", ")
            print("P = (%.3f, %.3f, %.3f)" %(com2[0], com2[1], com2[2]))
        # Start and stop timer with S
        elif e.key() == QtCore.Qt.Key_S:
            self.window.GLWidget.toggleTimer()
        elif e.key() == QtCore.Qt.Key_L:
            self.window.GLWidget.togglePlot()
        elif e.key() == QtCore.Qt.Key_G:
            self.window.GLWidget.toggleGrid()
        elif e.key() == QtCore.Qt.Key_N:
            self.window.GLWidget.toggleColors()
        elif e.key() == QtCore.Qt.Key_R:
            self.window.GLWidget.resetCenter()
        elif e.key() == QtCore.Qt.Key_E:
            E = self.window.GLWidget.oct.energy()
            J = self.window.GLWidget.oct.angularMomentum()
            print("E = %.4f, J = %.4f" %(E, J))
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