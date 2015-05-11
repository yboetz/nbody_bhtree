#include <iostream>
#include <math.h>
#include <xmmintrin.h>
#include <immintrin.h>


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
    
    f = _mm_set1_ps(f[0] + f[1] + f[2] + eps2);
   
    f = f*f*f;
    f = _mm_rsqrt_ps(f);
    f = _mm_mul_ps(m2, f);
    a = _mm_mul_ps(f, a);
    
    return a;
    }      
// Calculates potential between p1 and p2
float pot(__m128 p1, __m128 p2)
    {
    float m1 = p1[3];
    float m2 = p2[3];
    
    __m128 d = _mm_sub_ps(p2, p1);
    d = _mm_mul_ps(d, d);
       
    float f = d[0] + d[1] + d[2];
    f = m1 * m2 / sqrt(f);
    
    return f;
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
        
        Node(float*, float);
        Node(__m128);
        bool contains(__m128);
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
// Returns true if position at p is contained in Node
bool Node::contains(__m128 p)
    {
    __m128 d = midp - p;
    __m128 hside = _mm_set1_ps(midp[3] / 2.0f);
    __m128 c1 = _mm_cmplt_ps(d, hside);
    __m128 c2 = _mm_cmplt_ps(-hside, d);
    c1 = _mm_and_ps(c1, c2);
    
    return (c1[0] && c1[1] && c1[2]);
    }
// Returns integer value of suboctant of position p
short Node::whichOct(__m128 p)
    {
    __m128 _res = _mm_cmplt_ps(midp, p);
    _res = _mm_and_ps(_mm_set1_ps(1.0f), _res);
    short res = (short)(_res[0] + 2*_res[1] + 4*_res[2]);
    return res;    
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
        
        Cell(float*, float);
        Cell(__m128);
        void insert(__m128, int, Leaf**);
        void insertMultiple(float*, int, Leaf**); 
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
// Recursively inserts a body into cell
void Cell::insert(__m128 p, int n, Leaf** leavesptr)
    {
    short i = whichOct(p);
    Node* ptr = subp[i];
    
    if(ptr == NULL)                                             // If child does not exist, create leaf and insert body into leaf.
        {
        __m128 _side = _mm_set1_ps(midp[3] / 4.0f);             // Calculate midp of new leaf
        __m128 _midp = _mm_fmadd_ps(octIdx[i], _side, midp);
                  
        Leaf* _leaf = new Leaf (_midp);                         // Create new leaf
        subp[i] = (Node*)_leaf;                                 // Append ptr to leaf in list of subpointers
        leavesptr[n] = _leaf;                                   // Append ptr to leaf to a global list of all leaves  
                
        _leaf->com = p;                                         // Put pos and m into leaf
        }
    else if(ptr->type)                                          // If child == leaf, create new cell in place of leaf and insert both bodies in cell
        {
        Leaf* _leaf  = (Leaf*)subp[i];
        Cell* _cell = new Cell (_leaf->midp);                   // Create new cell in place of original leaf 
        subp[i] = (Node*)_cell;

        _cell->com = _leaf->com;
        
        short _i = _cell->whichOct(_leaf->com);                 // Calculates suboctand of original leaf
        _cell->subp[_i] = (Node*)_leaf;
        
        __m128 _side = _mm_set1_ps(_cell->midp[3] / 4.0f);      // Set parameters of leaf
        _leaf->midp = _mm_fmadd_ps(octIdx[_i], _side, _leaf->midp);  
                             
        _cell->insert(p, n, leavesptr);      
        }
    else ((Cell*)ptr)->insert(p, n, leavesptr);                 // If child == cell, recursively insert body into child
       
    __m128 _m = _mm_set1_ps(p[3]);                              // Set new mass & center of mass
    __m128 m = _mm_set1_ps(com[3]);
    __m128 M = _mm_add_ps(m, _m);
    
    __m128 _t = _mm_mul_ps(com, m);
    _t = _mm_fmadd_ps(p, _m, _t);
    
    com = _mm_div_ps(_t, M);
    com[3] = M[0];     
    }
// Calls insertion function for all N bodies in p
void Cell::insertMultiple(float* p, int N, Leaf** leavesptr)
    {
    for(int i = 0; i<N; i++)
        {
        __m128 _p = _mm_load_ps(p + 4*i);
        insert(_p, i, leavesptr);
        }
    }

// Octree class. Contains pointer to a root cell and a list of all leaves
class Octree
    {
    public:
        Node* root;
        Leaf** leaves;
        int N;
        float* pos;
        float* vel;
        float theta2;
        __m128 cent = {0.0f,0.0f,0.0f,0.0f};

        Octree(float*, float*, int, float);
        ~Octree();
        void buildTree();
        void getBoxSize();
        void delTree(Node*);                    // helper function for deconstructor
        float leafPot(Cell*, Leaf*);
        float energy();
        float angularMomentum();
        __m128 leafAccel(Cell*, Leaf*, float);
        void integrate(float, float);
        void integrateNSteps(float, float, int);
    };

// Constructor. Sets number of bodies, opening angle, center, creates list with leaves & builds first tree
Octree::Octree(float* p, float* v, int n, float th)
    {
    pos = p; 
    vel = v;
    N = n;
    theta2 = th;
    leaves = new Leaf*[N];
    buildTree();  
    }
// Recursively deletes every node and delete leaves
Octree::~Octree()
    {
    delTree(root);    
    delete[] leaves;
    }
// Recursively deletes every node 
void Octree::delTree(Node* node)
    {
    for(int i = 0; i < 8; i++)
        {
        Node* ptr = ((Cell*)node)->subp[i];
        if(ptr != NULL) 
            {
            if(ptr->type) delete (Leaf*)ptr;
            else delTree(ptr);
            }
        }
    delete (Cell*)node;
    }
// Finds boxsize around particles
void Octree::getBoxSize()
    {
    float side = 0;
   
    for(int i = 0; i < N; i++)
        {   
        int idx = 4*i;
        
        float s = fabs(pos[idx]);
        if(s > side) side = s;
        s = fabs(pos[idx + 1]);
        if(s > side) side = s;
        s = fabs(pos[idx + 2]);
        if(s > side) side = s;         
        }
        
    cent[3] = 2*side;
    }
// Creates new root cell and fills it with bodies
void Octree::buildTree()
    {
    getBoxSize();
    root = new Cell (cent);  
    ((Cell*)root)->insertMultiple(pos, N, leaves);
    cent = root->com;    
    }
// Traverses tree and calculates potential for a leaf
float Octree::leafPot(Cell* node, Leaf* leaf)
    {
    if(node == (Cell*)leaf) return 0.0f;
    
    if((node->type) || (pow((node->midp[3]),2) / dist(node->com, leaf->com) < theta2))
        return pot(leaf->com, node->com);
    else
        {
        float p = 0;
        for(int i = 0; i < 8; i++)
            {
            Cell* ptr = (Cell*)(node->subp[i]);
            if(ptr != NULL)
                {
                p += leafPot(ptr, leaf);
                }
            }
        return p;
        } 
    }
// Calculates energy of system (approximate)
float Octree::energy()
    {
    float V = 0;
    float T = 0;
    
    #pragma omp parallel for schedule(dynamic,100)
    for(int i = 0; i < N; i++)
        {
        int idx = 4*i;
        Leaf* leaf = leaves[i];
        
        #pragma omp atomic
        V += leafPot((Cell*)root, leaf);
        
        #pragma omp atomic
        T += (leaf->com[3]) * (vel[idx]*vel[idx] + vel[idx+1]*vel[idx+1] + vel[idx+2]*vel[idx+2]);
        }
        
    return 0.5f * (T - V);
    }
// Calculates angular momentum of system (exact)
float Octree::angularMomentum()
    {
    float J = 0;
    
    for(int i = 0; i < N; i++)
        {
        int idx = 4*i;
        
        __m128 p = _mm_load_ps(pos + idx);
        __m128 v = _mm_mul_ps(_mm_set1_ps(p[3]), _mm_load_ps(vel + idx));
        
        __m128 c = cross(p, v);    
        c = _mm_mul_ps(c, c);

        J += sqrt(c[0] + c[1] + c[2]);        
        }
        
    return J;
    }
// Traverses tree and calculates acceleration for a leaf
__m128 Octree::leafAccel(Cell* node, Leaf* leaf, float eps2)
    {
    if((node->type) || (pow((node->midp[3]),2) / dist(node->com, leaf->com) < theta2))
        return accel(leaf->com, node->com, eps2);
    else
        {
        __m128 a = {0.0f, 0.0f, 0.0f, 0.0f};
        for(int i = 0; i < 8; i++)
            {
            Cell* ptr = (Cell*)(node->subp[i]);
            if(ptr != NULL)
                {
                a = _mm_add_ps(a,leafAccel(ptr, leaf, eps2));
                }
            }
        return a;
        }
    }
// Finds acceleration for every leaf and updates pos & vel via semi-implizit Euler integration
void Octree::integrate(float dt, float eps2)
    {
    __m128 dtv = {dt, dt, dt, 0.0f};
    #pragma omp parallel for schedule(dynamic,100)
    for(int i = 0; i < N; i++)
        {
        int idx = 4*i;
        __m128 a = leafAccel((Cell*)root, leaves[i], eps2);
        __m128 p = _mm_load_ps(pos + idx);
        __m128 v = _mm_load_ps(vel + idx);
        
        v = _mm_fmadd_ps(dtv, a, v);
        p = _mm_fmadd_ps(dtv, v, p);
               
        _mm_store_ps(pos + idx, p);
        _mm_store_ps(vel + idx, v);
        }
    delTree(root);
    buildTree();
    }
// Calls integration function a number of times
void Octree::integrateNSteps(float dt, float eps2, int n)
    {
    for(int i = 0; i < n; i++)
        {
        integrate(dt, eps2);
        }
    }


// Recursively traverses tree and prints out level and node
void traverse(Node* node, int level)
    {
    std::cout << "Lvl " << level << ", "<< "Type = "<<((Cell*)node)->type << ", m = " << ((Cell*)node)->com[3] 
    << ", com = (" << ((Cell*)node)->com[0] << ", " << ((Cell*)node)->com[1] << ", " << ((Cell*)node)->com[2] 
    << "), midp = (" << ((Cell*)node)->midp[0] << ", " << ((Cell*)node)->midp[1] << ", " << ((Cell*)node)->midp[2] << "), Side = " << ((Cell*)node)->midp[3] << "\n";
    if(node->type == 0)
        {
        for(int i = 0; i < 8; i++)
            {
            if(((Cell*)node)->subp[i] != NULL) traverse(((Cell*)node)->subp[i], level+1);
            }
        }
    }
// Print all leaves
void prtleaves(Octree* root)
    {
    for(int i = 0; i < root->N; i++)
        {
        Leaf* ptr = root->leaves[i];
        std::cout << "Type = "<<ptr->type << ", com = (" << ptr->com[0] << ", " << ptr->com[1] << ", " << ptr->com[2] << ")\n";
        }
    }
