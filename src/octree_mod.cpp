#include <iostream>
#include <vector>
#include <x86intrin.h>
#include <omp.h>
#include "octree_mod.h"
#include "accel.cpp"
#include "utils.cpp"
#include "node.cpp"
#include <chrono>
using namespace std::chrono;

#pragma GCC diagnostic ignored "-Wignored-attributes"

// Constructor. Sets position, velocity, number of bodies, opening angle and softening length. Initializes Cell & Leaf vectors
Octree::Octree(float* p, float* v, int n,  int ncrit, float th, float e)
    {
    pos = p;
    vel = v;
    N = n;
    numCell = 0;
    Ncrit = ncrit;
    theta = th;
    eps = e;
    listCapacity = 0;
    T = 0;

    // timing
    T_accel = 0;
    T_insert = 0;
    T_walk = 0;

    leaves.resize(N);
    for(int i = 0; i < N; i++) leaves[i] = new Leaf(_mm_set1_ps(0.0f));

    root = makeCell(leaves[0]);
    root->com = _mm_set1_ps(0.0f);

    buildTree();
    }
// Destructor. Deletes every cell & leaf
Octree::~Octree()
    {
    const int cellsSize = cells.size();
    for(int i = 0 ; i < cellsSize; i++) delete cells[i];
    for(int i = 0; i < N; i++) delete leaves[i];
    }
// Returns pointer to cell. If there are enough cells in list, use one of those, if not, create new
Cell* Octree::makeCell(Leaf* leaf)
    {
    Cell* cell;
    if(numCell < (int)cells.size())
        {
        cell = cells[numCell];
        cell->midp = leaf->midp;
        }
    else
        {
        cell = new Cell (leaf->midp);
        cells.push_back(cell);
        }
    cell->com = leaf->com;
    for(int i = 0; i < 8; i++)
        cell->subp[i] = NULL;
    cell->n = 0;
    numCell++;
    return cell;
    }
// Prepares root for next iteration
void Octree::makeRoot()
    {
    numCell = 1;
    root->midp = root->com;
    root->com = _mm_set1_ps(0.0f);
    getBoxSize();
    Cell* _root = (Cell*)root;
    for(int i = 0; i < 8; i++)
        _root->subp[i] = NULL;
    _root->n = 0;
    }
// Recursively inserts a body into cell
void Octree::insert(Cell* cell, __m128 p, int id)
    {
    (cell->n)++;
    short i = cell->whichOct(p);
    Node* ptr = cell->subp[i];

    if(ptr == NULL)                                                     // If child does not exist, create leaf and insert body into leaf.
        {
        __m128 _side = _mm_permute_ps(cell->midp,0b11111111);           // Broadcast midp[3] to all elements of _side
        __m128 _midp = _mm_fmadd_ps(Cell::octIdx[i], _side, cell->midp);// Calculate midp of leaf with help of octIdx
                  
        Leaf* _leaf = leaves[id];                                       // Use existing leaf in list
        cell->subp[i] = (Node*)_leaf;                                   // Append ptr to leaf in list of subpointers

        _leaf->com = p;                                                 // Set centre of mass of leaf
        _leaf->midp = _midp;                                            // Set midpoint of leaf
        _leaf->id = id;                                                 // Set particle id in leaf
        }
    else if(ptr->type)                                                  // If child == leaf, create new cell in place of leaf and insert both bodies in cell
        {
        Leaf* _leaf  = (Leaf*)ptr;
        Cell* _cell = makeCell(_leaf);

        cell->subp[i] = (Node*)_cell;       
        
        short _i = _cell->whichOct(_leaf->com);                         // Calculates suboctant of original leaf in _cell
        _cell->subp[_i] = (Node*)_leaf;
        (_cell->n)++; 

        __m128 _side = _mm_permute_ps(_cell->midp,0b11111111);          // Broadcast midp[3] to all elements of _side
        _leaf->midp = _mm_fmadd_ps(Cell::octIdx[_i], _side,_leaf->midp);// Calculate midp of with help of octIdx
        
        insert(_cell, p, id);
        }
    else insert((Cell*)ptr, p, id);                                     // If child == cell, recursively insert body into child
    }
// Inserts all N bodies into root
void Octree::insertMultiple()
    {
    for(int i = 0; i<N; i++)
        {
        __m128 p = _mm_load_ps(pos + 4*i);
        insert((Cell*)root, p, i);
        }
    }
// Creates new root cell and fills it with bodies
void Octree::buildTree()
    {
    steady_clock::time_point t1_insert = steady_clock::now();
    makeRoot();
    insertMultiple();
    steady_clock::time_point t2_insert = steady_clock::now();
    steady_clock::time_point t1_walk = t2_insert;
    walkTree(root, root);
    getCrit();
    steady_clock::time_point t2_walk = steady_clock::now();
    T_insert += duration_cast<duration<double>>(t2_insert - t1_insert).count();
    T_walk += duration_cast<duration<double>>(t2_walk - t1_walk).count();
    }
/* Recursively walks tree and does three things:
    1. Threads tree for non-recursive tree walk (sets next & more pointers).
    2. Returns centre of mass of p to sum from bottom to top.
    3. Calculates distance of centre of mass and midpoint of cell.*/
__m128 Octree::walkTree(Node* p, Node* n)
    {
    p->next = n;

    if(p->type == 0)
        {
        Cell* ptr = (Cell*)p;
        __m128 com = _mm_set1_ps(0.0f);
        __m128 M = _mm_set1_ps(0.0f);

        int ndesc = 0;
        int i;
        Node* desc[9];
        ndesc = 0;
        for(i = 0; i < 8; i++)
            {
            Node* _ptr = ptr->subp[i];
            if(_ptr != NULL) desc[ndesc++] = _ptr;                  // if subcell exists, append to list 'desc'
            }

        ptr->more = desc[0];                                        // set 'more' pointer to first subcell
        desc[ndesc] = n;
        for(i = 0; i < ndesc; i++)                                  // loop over all subcells
            {
            __m128 _com = walkTree(desc[i], desc[i+1]);             // recursively call walkTree for all subcells
            __m128 _m = _mm_permute_ps(_com,0b11111111);

            com = _mm_fmadd_ps(_com,_m,com);                        // add com of subcell
            M = _mm_add_ps(M,_m);                                   // add up masses of subcells
            }
        com = _mm_div_ps(com,M);                                    // scale com with total mass of cell
        com = _mm_blend_ps(com,M,0b1000);                           // store total mass in com vector

        ptr->com = com;                                             // copy com to cell
        ptr->delta = dist(ptr->com, ptr->midp);                     // calculate distance between com and midpoint
        }

    return p->com;
    }
// Finds cells with less than Ncrit bodies and appends them to global list critCells
void Octree::getCrit()
    {
    critCells.resize(0);
    Node* node = root;
    do
        {       
        if(node->type == 0) 
            {
            if((((Cell*)node)->n) > Ncrit)
                node = ((Cell*)node)->more;
            else 
                {
                critCells.push_back(node);
                node = node->next;
                } 
            }
        else 
            {
            critCells.push_back(node);
            node = node->next;
            }
        }
    while(node != root);
    }
// Finds boxsize around particles
void Octree::getBoxSize()
    {
    __m128 side = _mm_set1_ps(0.0f);
    __m128 cent = root->midp;
    
    for(int i = 0; i < N; i++)
        {   
        __m128 p = _mm_sub_ps(_mm_load_ps(pos + 4*i), cent);
        p = _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)),p); // Absolute value
        side = _mm_max_ps(side,p);      
        }

    side = _mm_max_ps(side, _mm_permute_ps(side,_MM_SHUFFLE(3,1,0,2)));
    side = _mm_max_ps(side, _mm_permute_ps(side,_MM_SHUFFLE(3,1,0,2)));
    
    root->midp[3] = 2.0f*side[0];
    }
// Calculates energy of system (approximate)
float Octree::energy()
    {
    float V = 0;
    float T = 0;
    
    const int cSize = critCells.size();
    
    #pragma omp parallel
    {
    std::vector<__m128> list (listCapacity);
    float _V = 0;
    float _T = 0;
    
    #pragma omp for schedule(dynamic)
    for(int i = 0; i < cSize; i++)
        {
        list.resize(0);
        Cell* critCell = (Cell*)critCells[i];
        // Finds all cells which satisfy opening angle criterion and appends them to interaction list
        Node* node = root;
        do
            {
            if((node->type) || (((node->midp[3])/theta + ((Cell*)node)->delta) < cdist(critCell->midp, node->com)))
                {
                list.push_back(node->com);
                node = node->next;
                }
            else node = ((Cell*)node)->more;   
            }
        while(node != root);

        // If list is not __m256 aligned, adds another zero __m128 vector
        int listSize = list.size();
        if(listSize % 2 != 0)
            {
            list.push_back(_mm_set1_ps(0.0f));
            listSize++;
            }
        listSize/=2;
        float* lptr = (float*)&list[0];
        float* lptrStart = lptr;
        // For each leaf in critCell, calculate the energy by summing over all bodies in interaction list
        node = critCell;
        Node* end = critCell->next;
        do
            {
            if(node->type)
                {
                Leaf* leaf = (Leaf*)node;
                __m128 _p = leaf->com;
                __m256 _p1 = _mm256_set_m128(_p,_p);
                float p = 0.0f;

                for(int j = 0; j < listSize; j++)
                    {
                    __m256 _p2 = _mm256_loadu_ps(lptr);
                    p += pot(_p1,_p2);
                    lptr += 8;
                    }

                _V += p;

                __m128 v = _mm_load_ps(vel + 4*(leaf->id));
                v = _mm_dp_ps(v,v,0b01111111);
                _T += _p[3] * v[0];

                lptr = lptrStart;
                node = node->next;
                }
            else node = ((Cell*)node)->more;   
            }
        while(node != end);
        }
    #pragma omp atomic
    V += _V;
    
    #pragma omp atomic
    T += _T;
    
    #pragma omp single
    listCapacity = list.capacity();
    }

    __m128 mv = centreOfMomentum();
    mv = _mm_dp_ps(mv,mv,0b01111111);
    T -= (root->com[3]) * mv[0];

    return 0.5f * (T + V);
    }
// Calculates angular momentum of system (exact)
float Octree::angularMomentum()
    {
    __m128 J = _mm_set1_ps(0.0f);;
    __m128 mv = _mm_set1_ps(0.0f);;

    for(int i = 0; i < N; i++)
        {
        int idx = 4*i;
               
        __m128 p = _mm_load_ps(pos + idx);
        __m128 m = _mm_permute_ps(p,0b11111111);
        __m128 v = _mm_mul_ps(m, _mm_load_ps(vel + idx));
        
        J = _mm_add_ps(J, cross_ps(p, v));
      
        mv = _mm_add_ps(mv,v);
        }

    J = _mm_sub_ps(J,cross_ps(root->com,mv));
    J = _mm_dp_ps(J,J,0b01111111);
    J = _mm_sqrt_ps(J);

    return J[0];
    }
// Finds acceleration for every leaf and updates pos & vel via semi-implicit Euler integration. Rebuilds tree afterwards
void Octree::integrate(float dt)
    {
    steady_clock::time_point t1 = steady_clock::now();

    __m128 dtv = _mm_setr_ps(dt,dt,dt,0.0f);
    __m256 epsv = _mm256_set1_ps(eps);
    const int cSize = critCells.size();
    
    #pragma omp parallel
    {
    std::vector<__m128> list (listCapacity);
    
    #pragma omp for schedule(dynamic)
    for(int i = 0; i < cSize; i++)
        {
        list.resize(0);
        Cell* critCell = (Cell*)critCells[i];
        // Finds all cells which satisfy opening angle criterion and appends them to interaction list
        Node* node = root;
        do
            {
            if((node->type) || (((node->midp[3])/theta + ((Cell*)node)->delta) < cdist(critCell->midp, node->com)))
                {
                list.push_back(node->com);
                node = node->next;
                }
            else node = ((Cell*)node)->more;   
            }
        while(node != root);

        // If list is not __m256 aligned, adds another zero __m128 vector
        int listSize = list.size();
        if(listSize % 2 != 0)
            {
            list.push_back(_mm_set1_ps(0.0f));
            listSize++;
            }
        listSize/=2;
        float* lptr = (float*)&list[0];
        float* lptrStart = lptr;

        // For each leaf in critCell, calculate the acceleration by summing over all bodies in interaction list
        node = critCell;
        Node* end = critCell->next;
        do
            {
            if(node->type)
                {
                int idx = 4*(((Leaf*)node)->id);
                __m128 p = _mm_load_ps(pos + idx);
                __m128 v = _mm_load_ps(vel + idx);
                __m128 a = _mm_set1_ps(0.0f);
                __m256 _p1 = _mm256_set_m128(p,p);

                for(int j = 0; j < listSize; j++)
                    {
                    __m256 _p2 = _mm256_loadu_ps(lptr);
                    a = _mm_add_ps(a, accel(_p1, _p2, epsv));
                    lptr += 8;
                    }

                v = _mm_fmadd_ps(dtv, a, v);
                p = _mm_fmadd_ps(dtv, v, p);

                _mm_store_ps(pos + idx, p);
                _mm_store_ps(vel + idx, v);

                lptr = lptrStart;
                node = node->next;
                }
            else node = ((Cell*)node)->more;
            }
        while(node != end);
        }
    #pragma omp single
    listCapacity = list.capacity();
    }
    steady_clock::time_point t2 = steady_clock::now();
    T_accel += duration_cast<duration<double>>(t2 - t1).count();

    buildTree();
    T += dt;
    }
// Calls integration function a number of times
void Octree::integrateNSteps(float dt, int n)
    {
    for(int i = 0; i < n; i++)
        integrate(dt);
    }
// Returns centre of momentum
__m128 Octree::centreOfMomentum()
    {
    __m128 mv = _mm_set1_ps(0.0f);

    for(int i = 0; i < N; i++)
        {
        int idx = 4*i;
        __m128 m = _mm_set1_ps(pos[idx + 3]);
        __m128 v = _mm_mul_ps(m, _mm_load_ps(vel + idx));

        mv = _mm_add_ps(mv,v);
        }
    mv = _mm_div_ps(mv, _mm_set1_ps(root->com[3]));

    return mv;
    }
// Takes array of colors as argument and updates them
void Octree::updateColors(float* col)
    {
    __m128 one = _mm_set1_ps(1.1f);
    __m128 two = _mm_set1_ps(2.1f);
    for(int i = 0; i < N; i++)
        {
        int idx = 4*i;
        __m128 v = _mm_load_ps(vel + idx);

        __m128 _v = _mm_dp_ps(v,v,0b01111111);
        _v = _mm_rsqrt_ps(_v);

        v = _mm_fmadd_ps(v,_v,one);
        v = _mm_blend_ps(v,two,0b1000);
        v = _mm_div_ps(v,two);

        _mm_store_ps(col + idx, v);
        }
    }
// Takes array of linecolors and colors as argument and updates linecolors
void Octree::updateLineColors(float* col, float* linecol, int length)
    {
    for(int i = 0; i < N; i++)
        {
        for(int j = length-2; j > -1; j--)
            {
            int idx = 4*(i*length + j);
            _mm_store_ps(linecol + idx + 4, _mm_load_ps(linecol + idx));
            }
        int idx = 4*i;
        _mm_store_ps(linecol + idx*length, _mm_load_ps(col + idx));
        }
    }
// Takes array of linedata as argument and updates
void Octree::updateLineData(float* linedata, int length)
    {
    __m128i mask = _mm_set_epi32(0,-0x7fffffff,-0x7fffffff,-0x7fffffff);
    for(int i = 0; i < N; i++)
        {
        for(int j = length-2; j > -1; j--)
            {
            float* idx = linedata + 3*(i*length + j);
            __m128 p = _mm_loadu_ps(idx);
            _mm_maskstore_ps(idx + 3, mask, p);
            }
        __m128 p = _mm_load_ps(pos + 4*i);
        _mm_maskstore_ps(linedata + 3*i*length, mask, p);
        }
    }
// Saves centre of mass at position com
void Octree::saveCentreOfMass(float* com)
    {
    _mm_store_ps(com, root->com);
    }
// Saves centre of momentum at position com
void Octree::saveCentreOfMomentum(float* com)
    {
    _mm_store_ps(com, centreOfMomentum());
    }
