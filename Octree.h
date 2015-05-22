#include <iostream>
#include <math.h>
#include <vector>
#include <xmmintrin.h>
#include <immintrin.h>
#include <chrono>

// Calculates cross product of vectors a & b
__m128 cross(__m128 a, __m128 b)
    {
    __m128 res = _mm_sub_ps(
                            _mm_mul_ps(a, _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1))),
                            _mm_mul_ps(b, _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)))
                            );
    return _mm_shuffle_ps(res, res, _MM_SHUFFLE(3, 0, 2, 1 ));
    }
// Calculates acceleration between p1 and p2
__m128 accel(__m128 p1, __m128 p2, float eps2)
    {   
    __m128 a = _mm_sub_ps(p2, p1);
    __m128 m2 = _mm_set1_ps(p2[3]);
    
    __m128 f = _mm_mul_ps(a, a);
    
    f[3] = eps2;
    f = _mm_hadd_ps(f,f);
    f = _mm_hadd_ps(f,f);
   
    f = f*f*f;
    f = _mm_rsqrt_ps(f);
    f = _mm_mul_ps(m2, f);
    a = _mm_mul_ps(f, a);
    
    return a;
    }      
// Calculates potential between p1 and p2
float pot(__m128 p1, __m128 p2)
    {    
    __m128 d = _mm_sub_ps(p2, p1);
    d = _mm_mul_ps(d, d);
    d[3] = 0;
    d = _mm_hadd_ps(d,d);
    d = _mm_hadd_ps(d,d);
    d = _mm_rsqrt_ps(d);
    
    return p1[3]*p2[3]*d[0];
    }
// Returns squared distance between p1 & p2
float dist(__m128 p1, __m128 p2)
    {
    __m128 res = _mm_sub_ps(p2, p1);
    res = _mm_mul_ps(res, res);
    
    return res[0] + res[1] + res[2];   
    }
    
// Base class for both leaf & cell. Has all the basic definitions
class Node
    {  
    public:
        static __m128 octIdx[8];
        
        bool type;                                          // Type of node: Leaf == 1, Cell == 0
        __m128 midp;
        __m128 com = {0.0f,0.0f,0.0f,0.0f};
        Node* next = NULL;
        
        Node(float*, float);
        Node(__m128);
        short whichOct(__m128);      
    };
    
__m128 Node::octIdx[8] = {{-1,-1,-1,-2},{1,-1,-1,-2},{-1,1,-1,-2},{1,1,-1,-2},{-1,-1,1,-2},{1,-1,1,-2},{-1,1,1,-2},{1,1,1,-2}};

// Node constructor. Sets midpoint and side length
Node::Node(float* mp, float s)
    {
    midp[0] = mp[0];
    midp[1] = mp[1];
    midp[2] = mp[2];
    midp[3] = s;
    }
// Node constructor with float4 vector. Sets midpoint and side length
Node::Node(__m128 mp)
    {
    midp = mp;
    }
// Returns integer value of suboctant of position p
short Node::whichOct(__m128 p)
    {
    __m128 c = _mm_cmplt_ps(midp, p);
    c = _mm_and_ps(_mm_set1_ps(1.0f), c);
    short oct = (short)(c[0] + 2*c[1] + 4*c[2]);
    return oct;
    }
    
// Leaf: Class for a node without children & a single body within it
class Leaf: public Node
    {
    public: 
        Leaf(float*, float);
        Leaf(__m128);
    };
    
// Leaf constructor. Calls Node constructor & sets type
Leaf::Leaf(float* mp, float s) 
    : Node(mp, s)
    {
    type = 1;
    }
// Leaf constructor for SSE. Calls Node constructor & sets type
Leaf::Leaf(__m128 mp) 
    : Node(mp)
    {
    type = 1;
    }

// Cell: Class for a node with children & multiple bodies
class Cell: public Node
    {
    public:
        Node* subp[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};       // Pointer to children
        Node* more = NULL;                                                      // Pointer to first child
        float delta;
        
        Cell(float*, float);
        Cell(__m128);
    };
    
// Cell constructor. Calls Node constructor & sets type
Cell::Cell(float* mp, float s) 
    : Node(mp, s)
    {
    type = 0;
    }
// Cell constructor. Calls Node constructor & sets type
Cell::Cell(__m128 mp) 
    : Node(mp)
    {
    type = 0;
    }
    

// Octree class. Contains pointer to a root cell and a list of all leaves
class Octree
    {
    public:
        Node* root;
        std::vector<Leaf*> leaves;
        std::vector<Cell*> cells;
        int numCell;
        int N;
        float* pos;
        float* vel;
        float theta;
        float eps2;

        Octree(float*, float*, int, float, float);
        ~Octree();
        Cell* makeCell(Leaf*);
        void makeRoot();
        void insert(Cell*, __m128, int);
        void insertMultiple();
        void buildTree();
        void threadTree(Node*, Node*);
        void getBoxSize();
        float leafPot(Leaf*);
        float energy();
        float angularMomentum();
        __m128 leafAccel(Leaf*);
        void integrate(float);
        void integrateNSteps(float, int);
        __m128 centreOfMomentum();
    };
// Constructor. Sets position, velocity, number of bodies, opening angle and eps squared. Initializes Cell & Leaf vectors
Octree::Octree(float* p, float* v, int n, float th, float e2)
    {
    pos = p; 
    vel = v;
    N = n;
    theta = th;
    eps2 = e2;
        
    root = new Cell(_mm_set_ps(0.0f,0.0f,0.0f,0.0f));
    root->com = root->midp;
    
    cells.reserve((int)(1.1f * ((float)N) / 2.0f));
    cells.push_back((Cell*)root);
    
    leaves.resize(N);
    for(int i = 0; i < N; i++) leaves[i] = new Leaf(_mm_set_ps(0.0f,0.0f,0.0f,0.0f));
    
    buildTree();
    }
// Destructor. Deletes every cell & leaf
Octree::~Octree()
    {
    for(int i = 0 ; i < cells.size(); i++) delete cells[i];
    for(int i = 0; i < N; i++) delete leaves[i];
    }
// Returns pointer to cell. If there are enough cells in list, use one of those, if not, create new
Cell* Octree::makeCell(Leaf* leaf)
    {
    Cell* cell;
    if(numCell < cells.size()) 
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
    numCell++;
    return cell;
    }
// Prepares root for next iteration
void Octree::makeRoot()
    {
    numCell = 1;
    root->midp = root->com;
    root->com = _mm_set_ps(0.0f,0.0f,0.0f,0.0f);
    getBoxSize();
    for(int i = 0; i < 8; i++)
        ((Cell*)root)->subp[i] = NULL;
    }
// Recursively inserts a body into cell
void Octree::insert(Cell* cell, __m128 p, int n)
    {
    short i = cell->whichOct(p);
    Node* ptr = cell->subp[i];

    if(ptr == NULL)                                                     // If child does not exist, create leaf and insert body into leaf.
        {
        __m128 _side = _mm_set1_ps(cell->midp[3] / 4.0f);               // Calculate midp of new leaf
        __m128 _midp = _mm_fmadd_ps(Cell::octIdx[i], _side, cell->midp);
                  
        Leaf* _leaf = leaves[n];                                        // Use existing leaf in list
        cell->subp[i] = (Node*)_leaf;                                   // Append ptr to leaf in list of subpointers
                
        _leaf->com = p;                                                 // Put pos and m into leaf
        _leaf->midp = _midp;
        }
    else if(ptr->type)                                                  // If child == leaf, create new cell in place of leaf and insert both bodies in cell
        {
        Leaf* _leaf  = (Leaf*)ptr;
        Cell* _cell = makeCell(_leaf);

        cell->subp[i] = (Node*)_cell;        
        
        short _i = _cell->whichOct(_leaf->com);                         // Calculates suboctand of original leaf
        _cell->subp[_i] = (Node*)_leaf;

        __m128 _side = _mm_set1_ps(_cell->midp[3] / 4.0f);              // Set parameters of leaf
        _leaf->midp = _mm_fmadd_ps(Cell::octIdx[_i], _side, _leaf->midp);  
        
        insert(_cell, p, n);      
        }
    else insert((Cell*)ptr, p, n);                                      // If child == cell, recursively insert body into child
       
    __m128 _m = _mm_set1_ps(p[3]);                                      // Set new mass & center of mass
    __m128 m = _mm_set1_ps(cell->com[3]);
    __m128 M = _mm_add_ps(m, _m);
    
    __m128 _S = _mm_mul_ps(cell->com, m);
    _S = _mm_fmadd_ps(p, _m, _S);
    
    cell->com = _mm_div_ps(_S, M);
    cell->com[3] = M[0];
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
    makeRoot();
    insertMultiple();
    threadTree(root, root);
    }
// Threads tree for non-recursive walk. While doing also calculates distance of com to midp.
void Octree::threadTree(Node* p, Node* n)
    {
    p->next = n;
    if(p->type == 0)
        {
        Cell* ptr = (Cell*)p;
        ptr->delta = sqrt(dist(ptr->com, ptr->midp));               // Calculates distance between centre of mass and midpoint

        int ndesc = 0;
        int i;
        Node* desc[9];
        ndesc = 0;
        for(i = 0; i < 8; i++)
            {
            Node* _ptr = (Node*)(ptr->subp[i]);
            if(_ptr != NULL)
                {
                desc[ndesc++] = _ptr;
                }
            }
        ptr->more = desc[0];
        desc[ndesc] = n;
        for(i = 0; i < ndesc; i++)
            {
            threadTree(desc[i], desc[i+1]);
            }
        }
    }
// Finds boxsize around particles
void Octree::getBoxSize()
    {
    __m128 side = {0.0f,0.0f,0.0f,0.0f};
    __m128 cent = root->midp;
    
    for(int i = 0; i < N; i++)
        {   
        __m128 p = _mm_sub_ps(_mm_load_ps(pos + 4*i), cent);
        p = _mm_andnot_ps(_mm_castsi128_ps(_mm_set1_epi32(0x80000000)), p); // Absolute value
        side = _mm_max_ps(side,p);      
        }
    float s = side[0];
    if(s < side[1]) s = side[1];
    if(s < side[2]) s = side[2];
    
    root->midp[3] = 2*s;
    }
// Traverses tree and calculates potential for a leaf
float Octree::leafPot(Leaf* leaf)
    {   
    float p = 0.0f;
    
    Node* node = root;
    do
        {       
        if((node->type) || (pow((node->midp[3])/theta + ((Cell*)node)->delta,2) < dist(node->com, leaf->com)))
            {
            if((Leaf*)node != leaf) 
                p += pot(leaf->com, node->com);
            node = node->next;
            }
        else node = ((Cell*)node)->more;   
        }
    while(node != root);
    
    return p;
    }
// Calculates energy of system (approximate)
float Octree::energy()
    {
    float V = 0;
    float T = 0;
    
    #pragma omp parallel for schedule(dynamic,100)
    for(int i = 0; i < N; i++)
        {
        Leaf* leaf = leaves[i];
        
        #pragma omp atomic
        V += leafPot(leaf);
        
        __m128 v = _mm_load_ps(vel + 4*i);
        v = _mm_mul_ps(v,v);
        
        #pragma omp atomic
        T += (leaf->com[3]) * (v[0] + v[1] + v[2]);
        }

    __m128 mv = centreOfMomentum();
    mv = _mm_mul_ps(mv, mv);
    T -= (root->com[3]) * (mv[0] + mv[1] + mv[2]);

    return 0.5f * (T - V);
    }
// Calculates angular momentum of system (exact)
float Octree::angularMomentum()
    {
    __m128 J = {0.0f,0.0f,0.0f,0.0f};
    __m128 mv = {0.0f,0.0f,0.0f,0.0f};

    for(int i = 0; i < N; i++)
        {
        int idx = 4*i;
               
        __m128 p = _mm_load_ps(pos + idx);
        __m128 m = _mm_set1_ps(p[3]);
        __m128 v = _mm_mul_ps(m, _mm_load_ps(vel + idx));
        
        J = _mm_add_ps(J, cross(p, v));    
      
        mv = _mm_add_ps(mv,v);
        }
    
    J = _mm_sub_ps(J,cross(root->com,mv));
    J = _mm_mul_ps(J,J);    

    return sqrt(J[0] + J[1] + J[2]);
    }
// Traverses tree and calculates acceleration for a leaf
__m128 Octree::leafAccel(Leaf* leaf)
    {
    __m128 a = {0.0f,0.0f,0.0f,0.0f};
    
    Node* node = root;
    do
        {
        if((node->type) || (pow((node->midp[3])/theta + ((Cell*)node)->delta,2) < dist(node->com, leaf->com)))
            {
            a = _mm_add_ps(a, accel(leaf->com, node->com, eps2));
            node = node->next;
            }
        else node = ((Cell*)node)->more;   
        }
    while(node != root);
    
    return a;
    }
// Finds acceleration for every leaf and updates pos & vel via semi-implizit Euler integration
void Octree::integrate(float dt)
    {
    __m128 dtv = {dt,dt,dt,0.0f};
    #pragma omp parallel for schedule(dynamic,100)
    for(int i = 0; i < N; i++)
        {
        int idx = 4*i;
        __m128 a = leafAccel(leaves[i]);
        __m128 p = _mm_load_ps(pos + idx);
        __m128 v = _mm_load_ps(vel + idx);
        
        v = _mm_fmadd_ps(dtv, a, v);
        p = _mm_fmadd_ps(dtv, v, p);
               
        _mm_store_ps(pos + idx, p);
        _mm_store_ps(vel + idx, v);
        }
    buildTree();
    }
// Calls integration function a number of times
void Octree::integrateNSteps(float dt, int n)
    {
    for(int i = 0; i < n; i++)
        {
        integrate(dt);
        }
    }
// Returns centre of momentum
__m128 Octree::centreOfMomentum()
    {
    __m128 mv = {0.0f,0.0f,0.0f,0.0f};

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
