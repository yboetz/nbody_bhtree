#!/home/yboetz/.virtualenvs/pyznap/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:38:23 2015

@author: yboetz
"""

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from qtwindow import MainWindow


if __name__ == "__main__":
    # Start Qt applicatoin
    app = QtGui.QApplication([])
    # Create main window
    win = MainWindow()
    app.exec()
    pg.exit()
