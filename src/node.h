#ifndef NODE_H
#define NODE_H

#include <x86intrin.h>
#include "moments.h"

// Base class for both leaf & cell. Has all the basic definitions
class Node
    {  
    public:
        bool type;                  // type of node: Leaf == 1, Cell == 0
        __m128 com;                 // center of mass of node
        Node* next;                 // pointer to next node in threading
    };

// Leaf: Class for a node without children & a single body within it
class Leaf: public Node
    {
    public:
        int id;                     // particle id

        Leaf(int);
    };

// Cell: Class for a node with children & multiple bodies
class Cell: public Node
    {
    public:
        __m128 midp;                // midpoint (pos & sidelength) of node
        moment mom;                 // quadrupole tensor of node
        int n;                      // number of bodies inside cell
        float delta;                // distance from midpoint to centre of mass
        Node* subp[8];              // pointer to children
        Node* more;                 // pointer to first child
        static __m128 octIdx[8];

        Cell(__m128);
        short whichOct(__m128);
    };

// indexed vectors to easily calculate midpoint and sidelength of suboctants.
__m128 Cell::octIdx[8] = {{-0.25f,-0.25f,-0.25f,-0.5f},{0.25f,-0.25f,-0.25f,-0.5f},{-0.25f,0.25f,-0.25f,-0.5f},{0.25f,0.25f,-0.25f,-0.5f},{-0.25f,-0.25f,0.25f,-0.5f},{0.25f,-0.25f,0.25f,-0.5f},{-0.25f,0.25f,0.25f,-0.5f},{0.25f,0.25f,0.25f,-0.5f}};

#endif
