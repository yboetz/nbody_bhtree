# README #

This is an n-body simulation using a [Barnes-Hut](http://www.cita.utoronto.ca/~dubinski/treecode/treecode.html)
tree algorithm. Number crunching code is written in C++, leveraging OpenMP and SSE/AVX instructions
for high performance. This is then called in Python via Cython and visualized using pyqtgraph.
On an Intel i7-6700K one million particles can be simulated at more than one timestep per second.


#### Requirements ####

You need python 3.x with the following packages:

    Cython
    numpy
    pandas
    PyOpenGL
    PyOpenGL-accelerate
    PyQt5
    pyqtgraph
    h5py

I suggest installing [virtualenv & virtualenvwrapper](http://docs.python-guide.org/en/latest/dev/virtualenvs/),
so you don't clutter your system python installation with additional packages.


#### How do I set it up? ####

Clone the git repository

    git clone git@github.com:yboetz/nbody_bhtree.git

Then install the required python packages (best in your virtualenv)

    cd nbody_bhtree
    pip install -r requirements.txt

Compile the C++ code with

	make build

Finally start the widget

	python src/main.py


#### Key controls ####

Basic controls:

+ S - starts/stops the simulation
+ N - toggles colors of particles
+ L - toggles between dots and lines for particles
+ O - opens a window to load another data file
+ F - fullscreen
+ G - show cube
+ B - show full octree
+ P - cont. pan aroung center
+ Esc - close widget

Moving around:

+ Click & drag - rotate around center
+ Wheel or shift & drag - zoom in/out
+ Ctrl & drag - pan around
+ Q - recenter to center of mass

Simulation controls:

+ R - plot the energy and angular momentum evolution
+ E - print the energy and angular momentum to terminal
+ C - print the center of mass to terminal
+ T - evolve system a certain number of timesteps. Print duration and difference in energy and angular momentum to terminal
