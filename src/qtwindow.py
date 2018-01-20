# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 21:40:41 2018

@author: yboetz
"""

import os
import h5py
from math import ceil
from time import time
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from pandas import read_csv
from utils import centreOfMomentum, centreOfMass
from octree import OTree as Octree


class NBodyWidget(gl.GLViewWidget):
    """GL widget class to display nbody data. Stores the octree and all data about the simulation"""
    def __init__(self):
        super().__init__()

        # Timestep
        self.dt = np.float32(.005)
        # Softening length. CANNOT be zero or program will crash
        self.e = np.float32(.05)
        # Opening angle
        self.theta = np.float32(1)
        # Max number of bodies per critical cell
        self.Ncrit = 128
        # Tickrate. 1000/max framerate
        self.tickRate = 1000./60
        # Number of intermediate update steps before drawing
        self.burst = 1
        # Set distance to origin
        self.opts['distance'] = 35
        self.isPanning = False

        # Draw cube
        corners = [[-10, -10, -10], [-10, -10, 10], [-10, 10, 10], [-10, 10, -10], [10, -10, -10], [10, -10, 10], [10, 10, 10], [10, 10, -10]]
        cubedata = []
        for i in range(4):
            cubedata += [corners[i], corners[(i+1)%4], corners[i+4], corners[(i+1)%4+4], corners[i], corners[i+4]]
        self.cube = gl.GLLinePlotItem()
        self.cube.setData(pos=np.array(cubedata), antialias=True, mode='lines', width=0.5)

        # Initial size (position of size-slider)
        self.size = 75
        # Initial line length
        self.lineLength = 2
        # Set colors
        self.colors = (1, 1, .5, 1)
        self.lineColors = (1, 1, .5, 1)
        self.isColored = False

        # Create scatterplot with position data. Needs 3-vectors, self.pos is 4-aligned
        self.sp = gl.GLScatterPlotItem(pxMode=False)
        self.addItem(self.sp)
        # Create lineplot
        self.lp = gl.GLLinePlotItem()

        # Timer which calls update function at const framerate
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateData)
        self.timer.timeout.connect(self.updateScatterPlot)

        # Init fps counter
        self.fps = 1000 / self.tickRate
        self.timer.timeout.connect(self.fpsCounter)

        # Initial read in of positions and velocity from file. Creates octree
        self.dataPath = {'path': 'plummer', 'csv': False}
        self.readFile(**self.dataPath)

    # renderText has to be called inside paintGL
    def paintGL(self, *args, **kwds):
        super().paintGL(*args, **kwds)
        self.renderText(30, 30, "Fps:\t%.2f" %(self.fps))
        self.renderText(30, 45, "N:\t%i" %(self.n))
        self.renderText(30, 60, "Step:\t%.4f " %(self.dt))
        self.renderText(30, 75, "Time:\t%.3f" %(self.oct.T))

    # Pans around center at const rate of 2pi/min
    def cont_orbit(self):
        dp = 6 / self.fps
        self.orbit(dp,0)

    # Integrates positions & velocities self.burst steps forward
    def updateData(self):
        self.oct.integrateNSteps(self.dt, self.burst)

    # Updates scatterplot
    def updateScatterPlot(self):
        self.sp.setData(pos=self.pos.reshape((self.n, 4))[:,0:3],
                        size=self.sizeArray, color=self.colors)

    # Updates lineplot
    def updateLinePlot(self):
        self.oct.updateLineData(self.lineData, self.lineLength)
        self.lp.setData(pos=self.lineData, color=self.lineColors, antialias=True)

    # Setup of GLLinePlotItem
    def setupLinePlot(self):
        self.lineData = np.empty((self.n, self.lineLength, 3), dtype=np.float32)
        self.lineData[:,:,:] = self.pos.reshape((self.n,4))[:,None,0:3]
        self.lp.setData(pos=self.lineData, color=self.lineColors,
                        antialias=True, mode='lines', width=1.5)
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
                self.lineData = np.array(np.pad(self.lineData, ((0, 0), (0, length-self.lineLength),
                                         (0, 0)), 'edge'), dtype=np.float32)
                if self.isColored:
                    self.lineColors = np.array(np.pad(self.lineColors, ((0, 0), (0, length-self.lineLength),
                                               (0, 0)), 'edge'), dtype=np.float32)
            self.lp.setData(pos=self.lineData, color=self.lineColors, antialias=True)
        self.lineLength = length

    # Toggles paning
    def togglePan(self):
        if self.isPanning:
            self.timer.timeout.disconnect(self.cont_orbit)
            self.isPanning = False
        else:
            self.timer.timeout.connect(self.cont_orbit)
            self.isPanning = True

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
                self.lineColors = (1, 1, .5, 1)
            self.timer.timeout.connect(self.updateScatterPlot)
            self.addItem(self.sp)
            self.sp.setData(pos=self.pos.reshape((self.n, 4))[:,0:3],
                            size=self.sizeArray, color=self.colors)
        else:
            self.timer.timeout.disconnect(self.updateScatterPlot)
            self.removeItem(self.sp)
            self.setupLinePlot()

    # Toggle cube
    def toggleCube(self):
        if self.cube in self.items:
            self.removeItem(self.cube)
        else:
            self.addItem(self.cube)

    # Toggle colors
    def toggleColors(self):
        if self.isColored:
            self.timer.timeout.disconnect(self.updateColors)
            self.isColored = False
            self.colors = (1, 1, .5, 1)
            self.sp.setGLOptions('additive')
        else:
            self.timer.timeout.connect(self.updateColors)
            self.isColored = True
            self.colors = np.empty((self.n, 4), dtype=np.float32)
            self.updateColors()
            self.sp.setGLOptions('translucent')
        if self.sp in self.items:
            self.sp.setData(pos=self.pos.reshape((self.n, 4))[:,0:3],
                            size=self.sizeArray, color=self.colors)
        self.toggleLineColors()

    # Toggle colors of lines
    def toggleLineColors(self):
        if self.lp in self.items:
            if self.isColored:
                self.lineColors = np.ones((self.n, self.lineLength, 4), dtype=np.float32)
                self.lineColors[:,:,2] = .5
                self.updateLineColors()
                self.timer.timeout.connect(self.updateLineColors)
            else:
                self.timer.timeout.disconnect(self.updateLineColors)
                self.lineColors = (1, 1, .5, 1)
            self.lp.setData(pos=self.lineData, color=self.lineColors,
                            antialias=True)

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
            self.colors = (1, 1, .5, 1)
            self.sp.setGLOptions('additive')
            if self.lp in self.items:
                self.timer.timeout.disconnect(self.updateLineColors)
            self.lineColors = (1, 1, .5, 1)

    # Set dot size
    def setSize(self, size):
        self.size = size
        self.sizeArray = (size / 1000) * (self.pos[3::4] / np.amax(self.pos[3::4]))**(1/3)
        self.sp.setData(pos=self.pos.reshape((self.n, 4))[:,0:3],
                        size=self.sizeArray, color=self.colors)

    # Set dt
    def setdt(self, dt):
        self.dt = np.float32(dt / 10000)

    # Set burst
    def setBurst(self, burst):
        self.burst = burst

    # Reads in data from hdf5 or csv file
    def read(self, path, csv=False, num=8192):
        if csv:
            data = read_csv(path, delim_whitespace=True, header=None, dtype=np.float32,
                            keep_default_na=False)
            m = data.ix[:,0].values
            p = data.ix[:,1:3].values
            v = data.ix[:,4:6].values
        else:
            dirname = os.path.dirname(os.path.realpath(__file__))
            with h5py.File(os.path.join(dirname, '../data/data.h5'), 'r') as file:
                dset = file[path]
                m = dset['mass']
                p = dset['position']
                v = dset['velocity']
        # max number of bodies is given by len(m), min number is 1
        n = max(min(len(m), num), 1)
        pos = np.zeros((n, 4), dtype=np.float32)
        vel = np.zeros((n, 4), dtype=np.float32)
        # solar_system needs first 20 bodies (sun & planets)
        if 'solar_system' in path:
            if n >= 20:
                mask = np.concatenate((np.arange(0, 20), np.linspace(20, len(m), n - 20, endpoint=False, dtype=int)))
            else:
                mask = np.arange(0, n)
        else:
            mask = np.linspace(0, len(m), n, endpoint=False, dtype=int)
            m = m * len(m) / n # rescale masses for lower number of bodies
        # Mask out masses, positions & velocities
        pos[:,3] = m[mask]
        pos[:,:3] = p[mask]
        vel[:,:3] = v[mask]
        # Go to centre of mass & momentum system
        pos[:,:3] -= centreOfMass(pos)
        vel[:,:3] -= centreOfMomentum(vel, pos[:,3])
        # Copy data to class variables
        self.pos = pos.reshape(4*n)
        self.vel = vel.reshape(4*n)
        self.n = n

    # Reads a new file and sets up new octree class. Resets colors and plots
    def readFile(self, path, csv=True, num=8192):
        if self.timer.isActive():
            self.timer.stop()
        try:
            self.read(path, csv=csv, num=num)
        except OSError as error:
            print(error)
        except Exception as e:
            print(e)
            print('Read error. Data should be aligned as \'m x y z vx vy vz\'.')
        else:
            self.oct = Octree(self.pos, self.vel, self.n, self.Ncrit, self.theta, self.e)
            self.resetColors()
            self.setupRecording("del")
            self.setSize(self.size)
            self.resetCenter()
            self.lineData = None
            self.dataPath = {'path': path, 'csv': csv}
            if self.lp in self.items:
                self.togglePlot()
            if self.isPanning:
                self.togglePan()

    # Change number of bodies
    def changeNum(self, num):
        try:
            num = int(num)
        except ValueError:
            return
        if self.n == num:
            return
        else:
            self.readFile(num=num, **self.dataPath)

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
        super().mousePressEvent(ev)

    # Resets pan & zoom positon
    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self.prevZoomPos = None
        self.prevPanPos = None
        try:
            if self.panOnRelease:
                self.togglePan()
                del self.panOnRelease
        except AttributeError:
            pass

    # Pans in xy-plane
    def mouseMoveEvent(self, ev):
        """ Allow Shift to Move and Ctrl to Pan."""
        shift = ev.modifiers() == QtCore.Qt.ShiftModifier
        ctrl = ev.modifiers() == QtCore.Qt.ControlModifier
        if shift:
            y = ev.pos().y()
            if not hasattr(self, 'prevZoomPos') or not self.prevZoomPos:
                self.prevZoomPos = y
                return
            dy = y - self.prevZoomPos
            #ev.delta = lambda: -dy * 5
            ev.angleDelta = lambda: QtCore.QPoint(0, -dy * 5)
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
            super().mouseMoveEvent(ev)
            if self.isPanning:
                self.togglePan()
                self.panOnRelease = True

    # Makes a test: Calculates energy and momentum drift after num steps
    def test(self, dt, num):
        E0, J0 = self.oct.energy(), self.oct.angularMomentum()

        num_per_sec = 0.5 * 10**6
        appr_T = self.n * np.log(self.n) / num_per_sec / np.log(num_per_sec) * num
        if appr_T > 10:
            ok = input('{:d} steps might take a while. Continue y/n?\n'.format(num))
            if ok.lower() not in ['y', 'yes']:
                return

        print('Testing: ', end="")
        T = time()
        self.oct.integrateNSteps(np.float32(dt), num)
        T = time() - T

        E1, J1 = self.oct.energy(), self.oct.angularMomentum()
        dE, dJ = E1 - E0, J1 - J0
        print('Did {:d} cycles in {:.3f} seconds.'.format(num, T))
        print('E0 = {:.4f}, E1 = {:.4f}; J0 = {:.4f}, J1 = {:.4f}'.format(E0, E1, J0, J1))
        print('dE = {:.4f} = {:.4f} E0; dJ = {:.4f} = {:.4f} J0\n'.format(dE, dE / E0, dJ, dJ / J0))

    # Initial setup/destructor of recording
    def setupRecording(self, key="setup"):
        if key == "setup":
            pg.setConfigOptions(antialias=True)
#            pg.setConfigOption('background', 'w')
#            pg.setConfigOption('foreground', 'k')
            self.tData = np.empty(1000, dtype=np.float32)
            self.eData = np.empty(1000, dtype=np.float32)
            self.jData = np.empty(1000, dtype=np.float32)
            self.tData.fill(self.oct.T)
            self.eData.fill(self.oct.energy())
            self.jData.fill(self.oct.angularMomentum())
            self.GWin = pg.GraphicsWindow(title="Conserved quantities")
            self.GWin.resize(600, 600)
            self.GWin.ePlot = self.GWin.addPlot(title="Energy")
            self.GWin.nextRow()
            self.GWin.jPlot = self.GWin.addPlot(title="Angular momentum")
            self.ep = self.GWin.ePlot.plot(pen=(255, 0, 0), name="Energy")
            self.jp = self.GWin.jPlot.plot(pen=(0, 255, 0), name="Angular")
            self.GWin.ePlot.setMouseEnabled(False, False)
            self.GWin.jPlot.setMouseEnabled(False, False)
            self.isRecording = False
        elif key == "del":
            if hasattr(self,'GWin'):
                if self.isRecording:
                    self.toggleRecording()
                del self.GWin
                del self.tData, self.eData, self.jData
                del self.ep, self.jp

    # Starts/stops recording
    def toggleRecording(self):
        if not hasattr(self, 'GWin'):
            self.setupRecording()
        if self.isRecording:
            self.timer.timeout.disconnect(self.record)
            self.isRecording = False
            self.GWin.hide()
        else:
            self.timer.timeout.connect(self.record)
            self.isRecording = True
            self.frame = 0
            self.GWin.show()

    # Continuously updates energy & angular momentum plots
    def record(self):
        if not (self.frame % ceil(20/self.burst)):
            if self.GWin.isHidden():
                self.toggleRecording()
            else:
                self.tData = np.roll(self.tData, -1)
                self.eData = np.roll(self.eData, -1)
                self.jData = np.roll(self.jData, -1)
                self.tData[-1] = self.oct.T
                self.eData[-1] = self.oct.energy()
                self.jData[-1] = self.oct.angularMomentum()
                self.ep.setData(x=self.tData, y=self.eData)
                self.jp.setData(x=self.tData, y=self.jData)
        self.frame += 1


class Window(QtGui.QWidget):
    """QWidget class with controls & NBodyWidget"""
    def __init__(self):
        super().__init__()
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
        # Data list widget
        dataList = QtGui.QListWidget(self)
        path = os.path.dirname(os.path.realpath(__file__))
        with h5py.File(os.path.join(path, '../data/data.h5'), 'r') as file:
            for name, dset in file.items():
                item = QtGui.QListWidgetItem(name)
                dataList.addItem(item)
                if name == self.GLWidget.dataPath['path']:
                    dataList.setCurrentItem(item)
        dataListLabel = QtGui.QLabel('Datasets:', self)
        dataList.currentItemChanged.connect(lambda x: self.GLWidget.readFile(x.text(), csv=False,
                                            num=self.GLWidget.n))
        # Widget to st number of points
        numWidget = QtGui.QLineEdit(self)
        numWidget.setText('8192')
        numWidget.editingFinished.connect(lambda: self.GLWidget.changeNum(numWidget.text()))
        numWidgetLabel = QtGui.QLabel('Max. number of bodies:', self)
        # Labels for controls
        controlLabel = QtGui.QLabel('Controls:\nS\tStart/stop\nE\tPrint energy\n'
                                    'C\tPrint COM\nN\tToggle colors\nL\tToggle dots/lines\n'
                                    'O\tOpen file\nT\tTesting\nR\tPlot invariants\n'
                                    'Q\tReset center\nF\tFullscreen\nEsc\tClose\n\n'
                                    'Rotate\tClick&drag\nZoom\tWheel/Click&Shift\nPan\tClick&Ctrl', self)
        # Add widgets on grid
        grid.addWidget(self.GLWidget, 1, 3, 50, 50)
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
        grid.addWidget(numWidget, 15, 1, 1, 2)
        grid.addWidget(numWidgetLabel, 14, 1, 1, 2)
        grid.addWidget(dataList, 17, 1, 1, 2)
        grid.addWidget(dataListLabel, 16, 1, 1, 2)
        grid.addWidget(controlLabel, 20, 1, 1, 2)


class MainWindow(QtGui.QMainWindow):
    """Main window with file menu & statusbar"""
    def __init__(self):
        super().__init__()
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
        #openFile.setShortcut('O')
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
        # Defines which function to call at what keypress
        self.keyList = {
                        QtCore.Qt.Key_O: self.showDialog,
                        QtCore.Qt.Key_C: self.keyPressC,
                        QtCore.Qt.Key_S: self.window.GLWidget.toggleTimer,
                        QtCore.Qt.Key_L: self.window.GLWidget.togglePlot,
                        QtCore.Qt.Key_G: self.window.GLWidget.toggleCube,
                        QtCore.Qt.Key_N: self.window.GLWidget.toggleColors,
                        QtCore.Qt.Key_Q: self.window.GLWidget.resetCenter,
                        QtCore.Qt.Key_E: self.keyPressE,
                        QtCore.Qt.Key_R: self.window.GLWidget.toggleRecording,
                        QtCore.Qt.Key_T: self.keyPressT,
                        QtCore.Qt.Key_F: self.toggleFullScreen,
                        QtCore.Qt.Key_P: self.window.GLWidget.togglePan
                        }

    # Show file dialog and calls file read function
    def showDialog(self):
        if self.window.GLWidget.timer.isActive():
            self.window.GLWidget.timer.stop()
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open file','../data/')
        if path:
            self.window.GLWidget.readFile(path[0], csv=True, num=self.window.GLWidget.n)
    
    # Functions to call when key is pressed
    def keyPressC(self):
        com1 = self.window.GLWidget.oct.centreOfMass()
        com2 = self.window.GLWidget.oct.centreOfMomentum()
        print("R = ({:.3f}, {:.3f}, {:.3f})".format(*com1), end=", ")
        print("P = ({:.3f}, {:.3f}, {:.3f})".format(*com2))

    def keyPressE(self):
        E = self.window.GLWidget.oct.energy()
        J = self.window.GLWidget.oct.angularMomentum()
        print("E = {:.4f}, J = {:.4f}".format(E, J))

    def keyPressT(self):
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
    
    def toggleFullScreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def doNothing(self):
        pass

    # Calls function according to pressed key
    def keyPressEvent(self, e):
        if e.isAutoRepeat():
            pass
        else:
            self.keyList.get(e.key(), self.doNothing)()
