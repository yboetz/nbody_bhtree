#include <iostream>
#include <vector>
#include <x86intrin.h>
#include <omp.h>
#include "octree_mod.h"
#include "moments.cpp"
#include "accel.cpp"
#include "utils.cpp"
#include "node.cpp"
#include <chrono>
using namespace std::chrono;

#pragma GCC diagnostic ignored "-Wignored-attributes"

const int SIZEOF_COM = sizeof(__m128) / sizeof(float);      // sizeof com vector in floats
const int SIZEOF_MOM = sizeof(moment) / sizeof(float);      // sizeof moment struct in floats
const int SIZEOF_TOT = SIZEOF_COM + SIZEOF_MOM;

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
    // create all leaves at beginning so they don't have to be created at every step
    leaves.resize(N);
    for(int i = 0; i < N; i++) leaves[i] = new Leaf(i);
    // initialize root cell
    root = makeCell(_mm_set1_ps(0.0f));

    buildTree();
    }
// Destructor. Deletes every cell & leaf
Octree::~Octree()
    {
    for(int i = 0; i < (int)cells.size(); i++) delete cells[i];
    for(int i = 0; i < (int)leaves.size(); i++) delete leaves[i];
    }
// Returns pointer to cell. If there are enough cells in list, use one of those, if not, create new
Cell* Octree::makeCell(__m128 midp)
    {
    Cell* cell;
    if(numCell < (int)cells.size())
        {
        cell = cells[numCell];              // if there are enough cells in 'cells' vector, use one already created
        cell->midp = midp;                  // set mitpoint of cell
        }
    else
        {
        cell = new Cell (midp);             // if there are not yet enough cells, create a new one
        cells.push_back(cell);
        }
    // initialize com and quadrupole tensor
    cell->com = _mm_set1_ps(0.0f);
    moment_init(cell->mom);
    // set all children to NULL
    for(int i = 0; i < 8; i++)
        cell->subp[i] = NULL;
    cell->n = 0;                            // no bodies in cell at beginning
    numCell++;                              // total number of cells has increased by 1
    return cell;
    }
// Prepares root for next iteration
void Octree::makeRoot()
    {
    numCell = 0;
    root = makeCell(root->com);
    getBoxSize();
    }
// Recursively inserts a body into cell
void Octree::insert(Cell* cell, __m128 p, int id)
    {
    (cell->n)++;
    short i = cell->whichOct(p);
    Node* sub = cell->subp[i];

    if(sub == NULL)                                                 // If child does not exist, create leaf and insert body into leaf.
        {
        Leaf* leaf = leaves[id];                                    // Use existing leaf in list
        cell->subp[i] = (Node*)leaf;                                // Append leaf to list of subpointers
        leaf->com = p;                                              // com of leaf is position of body
        }
    else if(sub->type)                                              // If child == leaf, create new cell in place of leaf and insert both bodies in cell
        {
        __m128 midp = _mm_permute_ps(cell->midp, 0b11111111);       // Broadcast sidelength to all elements of midp
        midp = _mm_fmadd_ps(Cell::octIdx[i], midp, cell->midp);     // Calculate midp of new cell with help of octIdx
        Cell* _cell = makeCell(midp);
        cell->subp[i] = (Node*)_cell;
        
        short _i = _cell->whichOct(sub->com);                       // Calculates suboctant of original leaf in _cell
        _cell->subp[_i] = sub;                                      // Append leaf to list of subpointers
        (_cell->n)++;                                               // Increase count of bodies in _cell by one

        insert(_cell, p, id);                                       // Recursively insert leaf into new cell
        }
    else insert((Cell*)sub, p, id);                                 // If child == cell, recursively insert body into child
    }
// Inserts all N bodies into root
void Octree::insertMultiple()
    {
    for(int i = 0; i<N; i++)
        {
        __m128 p = _mm_load_ps(pos + 4*i);                          // load next body
        insert(root, p, i);                                         // insert into root
        }
    }
// Creates new root cell and fills it with bodies
void Octree::buildTree()
    {
    steady_clock::time_point t1_insert = steady_clock::now();
    makeRoot();                                                     // prepare root for new tree
    insertMultiple();                                               // insert all bodies into the root node
    steady_clock::time_point t2_insert = steady_clock::now();
    steady_clock::time_point t1_walk = t2_insert;
    walkTree(root, root);                                           // recursively walk tree
    getCrit();                                                      // store all critical cells (n < Ncrit) into list
    steady_clock::time_point t2_walk = steady_clock::now();
    T_insert += duration_cast<duration<double>>(t2_insert - t1_insert).count();
    T_walk += duration_cast<duration<double>>(t2_walk - t1_walk).count();
    }
/* Recursively walks tree and does three things:
    1. Threads tree for non-recursive tree walk (sets next & more pointers)
    2. Sets center of mass and quadrupole moment tensor
    3. Calculates distance of centre of mass and midpoint of cell*/
void Octree::walkTree(Node* p, Node* n)
    {
    p->next = n;                                                    // 'next' points to next cell on same level or first on upper level

    if(p->type == 0)                                                // if type == cell
        {
        __m128 M = _mm_set1_ps(0.0f);
        int ndesc = 0;
        Node* desc[9];
        ndesc = 0;
        for(int i = 0; i < 8; i++)                                  // loop over all child nodes
            {
            Node* sub = ((Cell*)p)->subp[i];
            if(sub != NULL) desc[ndesc++] = sub;                    // if child exists, append to list 'desc'
            }

        ((Cell*)p)->more = desc[0];                                 // set 'more' pointer to first subcell
        desc[ndesc] = n;
        for(int i = 0; i < ndesc; i++)
            {
            walkTree(desc[i], desc[i+1]);                           // recursively call walkTree for all subcells
            // calculate com
            __m128 m = _mm_permute_ps(desc[i]->com, 0b11111111);    // m = mass of child node
            p->com = _mm_fmadd_ps(desc[i]->com, m, p->com);         // add center of mass of child node
            M = _mm_add_ps(M, m);                                   // sum up total mass in children
            }
        p->com = _mm_div_ps(p->com, M);                             // divide calculated center of mass by total mass
        p->com = _mm_blend_ps(p->com, M, 0b1000);                   // store total mass in com vector
        ((Cell*)p)->delta = dist(p->com, ((Cell*)p)->midp);         // calculate distance between com and midpoint
        // compute quadrupole tensor
        for(int i = 0; i < ndesc; i++)
            moment_add_sub((Cell*)p, desc[i]);                      // add quad. moment uf subcell
        }
    }
// Finds cells with less than Ncrit bodies and appends them to global list critCells
void Octree::getCrit()
    {
    critCells.resize(0);                                            // initialize critCells list
    Node* node = root;                                              // start at root
    do
        {       
        if(node->type == 0)                                         // if type == cell
            {
            if((((Cell*)node)->n) > Ncrit)                          // if number of bodies > Ncrit
                node = ((Cell*)node)->more;                         // go one level deeper
            else 
                {
                critCells.push_back(node);                          // if not, add cell to list of critCells
                node = node->next;                                  // continue on same or higher level
                } 
            }
        else 
            {
            critCells.push_back(node);                              // if type == leaf, we cannot go deeper
            node = node->next;                                      // so add leaf to list of critCells
            }
        }
    while(node != root);
    }
// Finds boxsize around particles
void Octree::getBoxSize()
    {
    __m128 side = _mm_set1_ps(0.0f);
    __m128 cent = root->midp;
    
    for(int i = 0; i < N; i++)                                      // iterate over all bodies
        {   
        __m128 p = _mm_sub_ps(_mm_load_ps(pos + 4*i), cent);        // distance to center
        p = _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)),p); // absolute value
        side = _mm_max_ps(side,p);                                  // find maximal value
        }
    // find maximal value over first 3 entries
    side = _mm_max_ps(side, _mm_permute_ps(side,_MM_SHUFFLE(3,1,0,2)));
    side = _mm_max_ps(side, _mm_permute_ps(side,_MM_SHUFFLE(3,1,0,2)));

    root->midp[3] = 2.0f*side[0];                                   // sidelength of root node is 2x max. norm of farthest particle
    }
// Calculates energy of system (approximate)
float Octree::energy()
    {
    const __m128 ZERO = _mm_set1_ps(0.0f);
    __m256 epsv = _mm256_set1_ps(eps);
    float V = 0;                                                    // total potential energy
    float T = 0;                                                    // total kinetic energy
    
    #pragma omp parallel
    {
    // in parallel region initialice temporary interaction lists and energy variables for each thread
    std::vector<float> int_cells (listCapacity);
    std::vector<float> int_leaves (4*Ncrit);
    float _V = 0.0f;
    float _T = 0.0f;

    #pragma omp for schedule(dynamic)
    for(int i = 0; i < (int)critCells.size(); i++)                  // loop over all critical cells
        {
        int_cells.resize(0);
        int_leaves.resize(0);
        Cell* critCell = (Cell*)critCells[i];
        // Finds all cells which satisfy opening angle criterion and appends them to interaction list
        Node* node = root;
        do
        {
            Cell* _node = (Cell*)node;
            const float* begin_com = (float*)&(node->com);          // pointer to first element of center of mass
            const float* begin_mom = (float*)&(((Cell*)node)->mom); // only defined if type of node == cell
            if(node->type)
            {
                int_leaves.insert(int_leaves.end(), begin_com, begin_com + SIZEOF_COM); // append center of mass vector
                node = node->next;
            }
            // if critCell == leaf we have to calculate cdist using com instead of midp
            else if((critCell->type == 0 && ((_node->midp[3])/theta + _node->delta) < cdist(critCell->midp, _node->com)) ||
                    (critCell->type == 1 && ((_node->midp[3])/theta + _node->delta) < cdist(_mm_blend_ps(critCell->com, ZERO, 0b1000), _node->com)))
            {
                int_cells.insert(int_cells.end(), begin_com, begin_com + SIZEOF_COM);   // append center of mass vector
                int_cells.insert(int_cells.end(), begin_mom, begin_mom + SIZEOF_MOM);   // append moment tensor struct
                node = node->next;
            }
            else node = _node->more;   
        }
        while(node != root);

        // if list are not 2-aligned, add another entry
        if(int_leaves.size() % (2*SIZEOF_COM) != 0)
            int_leaves.insert(int_leaves.end(), SIZEOF_COM, 0.0f);
        if(int_cells.size() % (2*SIZEOF_TOT) != 0)
            int_cells.insert(int_cells.end(), SIZEOF_TOT, 0.0f);

        // For each leaf in critCell, calculate the energy by summing over all bodies in interaction list
        node = critCell;
        Node* end = critCell->next;
        do
            {
            if(node->type)
                {
                float* cells_ptr = int_cells.data();                // pointer to first element in interaction list (better performance)
                int idx = 4*(((Leaf*)node)->id);
                __m128 p = _mm_load_ps(pos + idx);
                __m128 v = _mm_load_ps(vel + idx);
                __m256 _p1 = _mm256_set_m128(p, p);
                float V_tmp = 0.0f;                                 // is needed else result is not accurate?

                for(int j = 0; j < (int)int_leaves.size(); j+=2*SIZEOF_COM)
                {
                    __m256 _p2 = _mm256_loadu_ps(int_leaves.data() + j);    // read in com of 2 particles
                    V_tmp += pot(_p1, _p2, epsv);                           // calculate potential
                }
                for(int j = 0; j < (int)int_cells.size(); j+=2*SIZEOF_TOT)
                {
                    __m256 _p2 = _mm256_loadu2_m128(cells_ptr, cells_ptr+SIZEOF_TOT);   // read in com of 2 particles
                    __m256 _q1 = _mm256_loadu2_m128(cells_ptr+SIZEOF_COM, cells_ptr+SIZEOF_TOT+SIZEOF_COM);     // read in quad. tensor
                    __m256 _q2 = _mm256_loadu2_m128(cells_ptr+SIZEOF_COM+2, cells_ptr+SIZEOF_TOT+SIZEOF_COM+2); // read in quad. tensor
                    V_tmp += pot(_p1, _p2, _q1, _q2, epsv);         // calculate potential
                    cells_ptr += 2*SIZEOF_TOT;                      // increment pointer by 2
                }
                _V += V_tmp;
                // calculate kinetic energy
                v = _mm_dp_ps(v, v, 0b01111111);
                _T += p[3] * v[0];

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
    listCapacity = int_cells.capacity();
    }
    // subtract kinetic energy of center of mass
    __m128 mv = centreOfMomentum();
    mv = _mm_dp_ps(mv, mv, 0b01111111);
    T -= (root->com[3]) * mv[0];

    return 0.5f * (T - V);
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

    J = _mm_sub_ps(J, cross_ps(root->com, mv));
    J = _mm_dp_ps(J, J, 0b01111111);
    J = _mm_sqrt_ps(J);

    return J[0];
    }
// Finds acceleration for every leaf and updates pos & vel via semi-implicit Euler integration. Rebuilds tree afterwards
void Octree::integrate(float dt)
    {
    steady_clock::time_point t1 = steady_clock::now();

    __m128 dtv = _mm_setr_ps(dt, dt, dt, 0.0f);
    __m256 epsv = _mm256_set1_ps(eps);
    
    #pragma omp parallel
    {
    // in parallel region initialice temporary interaction lists for each thread
    std::vector<float> int_cells (listCapacity);
    std::vector<float> int_leaves (4*Ncrit);
    
    #pragma omp for schedule(dynamic)
    for(int i = 0; i < (int)critCells.size(); i++)                  // loop over all critical cells
        {
        int_cells.resize(0);
        int_leaves.resize(0);
        Cell* critCell = (Cell*)critCells[i];
        // Finds all cells which satisfy opening angle criterion and appends them to interaction list
        Node* node = root;
        do
        {
            Cell* _node = (Cell*)node;
            const float* begin_com = (float*)&(node->com);          // pointer to first element of center of mass
            const float* begin_mom = (float*)&(((Cell*)node)->mom); // only defined if type of node == cell
            if(node->type)
            {
                int_leaves.insert(int_leaves.end(), begin_com, begin_com + SIZEOF_COM); // append center of mass vector
                node = node->next;
            }
            // if critCell == leaf we have to calculate cdist using com instead of midp
            else if((critCell->type == 0 && ((_node->midp[3])/theta + _node->delta) < cdist(critCell->midp, _node->com)) ||
                    (critCell->type == 1 && ((_node->midp[3])/theta + _node->delta) < cdist(_mm_blend_ps(critCell->com, dtv, 0b1000), _node->com)))
            {
                int_cells.insert(int_cells.end(), begin_com, begin_com + SIZEOF_COM);   // append center of mass vector
                int_cells.insert(int_cells.end(), begin_mom, begin_mom + SIZEOF_MOM);   // append moment tensor struct
                node = node->next;
            }
            else node = _node->more;   
        }
        while(node != root);

        // if list are not 2-aligned, add another entry
        if(int_leaves.size() % (2*SIZEOF_COM) != 0)
            int_leaves.insert(int_leaves.end(), SIZEOF_COM, 0.0f);
        if(int_cells.size() % (2*SIZEOF_TOT) != 0)
            int_cells.insert(int_cells.end(), SIZEOF_TOT, 0.0f);

        // For each leaf in critCell, calculate the acceleration by summing over all bodies in interaction list
        node = critCell;
        Node* end = critCell->next;
        do
            {
            if(node->type)
                {
                float* cells_ptr = int_cells.data();                // pointer to first element in interaction list (better performance)
                int idx = 4*(((Leaf*)node)->id);
                __m128 p = _mm_load_ps(pos + idx);
                __m128 v = _mm_load_ps(vel + idx);
                __m128 a = _mm_set1_ps(0.0f);
                __m256 _p1 = _mm256_set_m128(p, p);

                for(int j = 0; j < (int)int_leaves.size(); j+=2*SIZEOF_COM)
                {
                    __m256 _p2 = _mm256_loadu_ps(int_leaves.data() + j);    // read in com of 2 particles
                    a = _mm_add_ps(a, accel(_p1, _p2, epsv));               // calculate acceleration
                }
                for(int j = 0; j < (int)int_cells.size(); j+=2*SIZEOF_TOT)
                {
                    __m256 _p2 = _mm256_loadu2_m128(cells_ptr, cells_ptr+SIZEOF_TOT);   // read in com of 2 particles
                    __m256 _q1 = _mm256_loadu2_m128(cells_ptr+SIZEOF_COM, cells_ptr+SIZEOF_TOT+SIZEOF_COM);     // read in quad. tensor
                    __m256 _q2 = _mm256_loadu2_m128(cells_ptr+SIZEOF_COM+2, cells_ptr+SIZEOF_TOT+SIZEOF_COM+2); // read in quad. tensor
                    a = _mm_add_ps(a, accel(_p1, _p2, _q1, _q2, epsv));     // calculate acceleration
                    cells_ptr += 2*SIZEOF_TOT;                              // increment pointer by 2
                }
                // semi-implicit Euler integration
                v = _mm_fmadd_ps(dtv, a, v);
                p = _mm_fmadd_ps(dtv, v, p);
                // store new position & velocity
                _mm_store_ps(pos + idx, p);
                _mm_store_ps(vel + idx, v);

                node = node->next;
                }
            else node = ((Cell*)node)->more;
            }
        while(node != end);
        }
    #pragma omp single
    listCapacity = int_cells.capacity();
    }
    steady_clock::time_point t2 = steady_clock::now();
    T_accel += duration_cast<duration<double>>(t2 - t1).count();
    // rebuild tree
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
// saves midpoints of all cells in list
void Octree::save_midp(float* list)
{
    for(int i = 0; i < numCell; i++)
        _mm_store_ps(list + 4*i, cells[i]->midp);
}