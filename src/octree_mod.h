#include <vector>
#include <x86intrin.h>
#include "moments.h"
#include "node.h"
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif
#ifndef OCTREE_MOD_H
#define OCTREE_MOD_H

// Octree class
class Octree
    {
    public:
        Cell* root;                                         // root cell
        std::vector<Leaf*> leaves;                          // list of all leaves
        std::vector<Cell*> cells;                           // list of all cells
        std::vector<Node*> critCells;                       // list of all critical cells
        int numCell;                                        // total number of cells
        int N;                                              // total number of bodies
        int Ncrit;                                          // max. number of bodies per critical cell
        float* pos;                                         // pointer to array containing positions (x, y, z, m)
        float* vel;                                         // pointer to array containing velocities (vx, vy, vz, 0)
        float theta;                                        // opening angle
        int step;                                           // current step
        double T;                                           // time passed in system (not physical units)
        cl::Context context;
        cl::CommandQueue queue;
        cl::Program program;

        double T_insert, T_accel, T_walk;                   // variables for time measurements

        Octree(float*, float*, int, int, float);
        ~Octree();
        Cell* makeCell(__m128);
        void makeRoot();
        void insert(Cell*, __m128, int);
        void insertMultiple();
        void buildTree();
        void walkTree(Node*, Node*);
        void getCrit();
        void getBoxSize();
        float energy();
        float angularMomentum();
        void integrate(float);
        void integrateNSteps(float, int);
        __m128 centreOfMomentum();
        void updateColors(float*);
        void updateLineColors(float*, float*, int);
        void updateLineData(float*, int);
        void saveCentreOfMass(float*);
        void saveCentreOfMomentum(float*);
        void save_midp(float*);
    };

#endif
