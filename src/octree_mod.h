#include <vector>
#include <x86intrin.h>
#include "node.h"
#ifndef OCTREE_MOD_H
#define OCTREE_MOD_H

// Octree class
class Octree
    {
    public:
        Node* root;
        std::vector<Leaf*> leaves;
        std::vector<Cell*> cells;
        std::vector<Node*> critCells;
        int listCapacity;
        int numCell;
        int N;
        int Ncrit;
        float* pos;
        float* vel;
        float theta;
        float eps;
        double T;
        int numThreads;

        double T_insert, T_accel, T_walk;

        Octree(float*, float*, int, int, float, float);
        ~Octree();
        Cell* makeCell(Leaf*);
        void makeRoot();
        void insert(Cell*, __m128, int);
        void insertMultiple();
        void buildTree();
        __m128 walkTree(Node*, Node*);
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
    };

#endif
