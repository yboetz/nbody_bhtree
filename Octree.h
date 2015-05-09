#include <iostream>
#include <algorithm>


// Calculates acceleration between p1 and p2 and stores it in a
void accel(float* p1, float* p2, float* a, float m2, float eps2)
    {
    a[0] = p2[0] - p1[0];
    a[1] = p2[1] - p1[1];
    a[2] = p2[2] - p1[2];
       
    float f = a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + eps2;
    f = 1.0f/sqrt(f);
    f = m2 * f * f * f;
    
    a[0] *= f;
    a[1] *= f;
    a[2] *= f;
    }
// Calculates potential between p1 and p2
float pot(float* p1, float* p2, float m1, float m2)
    {
    float d[3];
    d[0] = p2[0] - p1[0];
    d[1] = p2[1] - p1[1];
    d[2] = p2[2] - p1[2];
       
    float f = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    f = m1 * m2 / sqrt(f);
    
    return f;
    }
// Returns squared distance between p1 & p2
float dist(float* p1, float* p2)
    {
    float res[3];
    res[0] = p2[0] - p1[0];
    res[1] = p2[1] - p1[1];
    res[2] = p2[2] - p1[2];
    
    return res[0]*res[0] + res[1]*res[1] + res[2]*res[2];    
    }
    
// Base class for both leaf & cell. Has all the basic definitions
class Node
    {  
    public:
        static float octIdx[8][3];
        
        bool type;                                          // Type of node: Leaf == 1, Cell == 0
        float side, m;
        float midp[3];
        float com[3] = {0, 0, 0};
        
        Node(float*, float);
        bool contains(float*);
        short whichOct(float*);      
    };
    
float Node::octIdx[8][3] = {{-1,-1,-1},{1,-1,-1},{-1,1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,1},{1,1,1}};

// Node constructor. Sets midpoint and side length
Node::Node(float* mp, float s)
    {
    midp[0] = mp[0];
    midp[1] = mp[1];
    midp[2] = mp[2];
    side = s;
    m = 0;
    }
// Returns true if position at p is contained in Node
bool Node::contains(float* p)
    {
    float d[3];
    d[0] = (midp)[0] - p[0];
    d[1] = (midp)[1] - p[1];
    d[2] = (midp)[2] - p[2];
    float hside = side / 2.0f;
    
    if(fabs(d[0]) < hside && fabs(d[1]) < hside && fabs(d[2]) < hside) return 1;
    else return 0;
    }
// Returns integer value of suboctant of position p
short Node::whichOct(float* p)
    {
    short res = 0;
    res += (midp[0] < p[0]);
    res += 2 * (midp[1] < p[1]);
    res += 4 * (midp[2] < p[2]);
    return res;    
    }
    
// Leaf: Class for a node without children & a single body within it
class Leaf: public Node
    {
    public: 
        Leaf(float*, float);
    };
    
// Leaf constructor. Calls Node constructor & sets type
Leaf::Leaf(float* mp, float s) 
    : Node(mp, s)
    {
    type = 1;
    }

// Cell: Class for a node with children & multiple bodies
class Cell: public Node
    {
    public:
        Node* subp[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};       // Pointer to children
        
        Cell(float*, float);
        void insert(float*, int, Leaf**);
        void insertMultiple(float*, int, Leaf**);
    
    };
    
// Cell constructor. Calls Node constructor & sets type
Cell::Cell(float* mp, float s) 
    : Node(mp, s)
    {
    type = 0;
    }

// Recursively inserts a body into cell
void Cell::insert(float* p, int n, Leaf** leavesptr)
    {
    short i = whichOct(p);
    
    if(subp[i] == NULL)                                         // If child does not exist, create leaf and insert body into leaf.
        {
        float _midp[3];
        float _side = side / 4.0f;
        _midp[0] = midp[0] + octIdx[i][0] * _side;
        _midp[1] = midp[1] + octIdx[i][1] * _side;
        _midp[2] = midp[2] + octIdx[i][2] * _side;
        
        Leaf* _leaf = new Leaf (_midp, 2.0f * _side);           // create new leaf
        subp[i] = (Node*)_leaf;                                 // append ptr to leaf in list of subpointers
        leavesptr[n] = _leaf;                                   // Append ptr to leaf to a global list of all leaves  
                
        _leaf->m = p[3];                                        // Put pos and m into leaf
        _leaf->com[0] = p[0];
        _leaf->com[1] = p[1];
        _leaf->com[2] = p[2];

        }
    else if(subp[i]->type)                                      // If child == leaf, create new cell in place of leaf and insert both bodies in cell
        {
        Leaf* _leaf  = (Leaf*)subp[i];
        Cell* _cell = new Cell (_leaf->midp, _leaf->side);      // Create new cell in place of original leaf 
        subp[i] = (Node*)_cell;

        _cell->m = _leaf->m;
        _cell->com[0] = _leaf->com[0];
        _cell->com[1] = _leaf->com[1];
        _cell->com[2] = _leaf->com[2];
        
        short _i = _cell->whichOct(_leaf->com);                // Calculates suboctand of original leaf
        _cell->subp[_i] = (Node*)_leaf;
        
        // Set parameters of leaf
        float _side = (_cell->side) / 4.0f;
        _leaf->midp[0] += octIdx[_i][0] * _side;
        _leaf->midp[1] += octIdx[_i][1] * _side;
        _leaf->midp[2] += octIdx[_i][2] * _side;
        _leaf->side = 2.0f * _side;
        
        _cell->insert(p, n, leavesptr);
        
        }
    else ((Cell*)subp[i])->insert(p, n, leavesptr);             // If child == cell, recursively insert body into child
    
    // Set new mass & center of mass        
    float _m = p[3];
    float M = m + _m;
    com[0] = (com[0] * m + p[0] * _m) / M;
    com[1] = (com[1] * m + p[1] * _m) / M;
    com[2] = (com[2] * m + p[2] * _m) / M;
    m = M;    
    }
// Calls insertion function for all N bodies in p
void Cell::insertMultiple(float* p, int N, Leaf** leavesptr)
    {
    for(int i = 0; i<N; i++)
        {
        insert(p + 4*i, i, leavesptr);
        }
    }

void traverse(Node*, int); // Define function so it can be called in class Octree

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
        float cent[3];

        Octree(float*, float*, int, float*, float);
        ~Octree();
        void buildTree();
        float getBoxSize();
        void delTree(Node*);                    // helper function for deconstructor
        float leafPot(Cell*, Leaf*);
        float energy();
        void leafAccel(Cell*, Leaf*, float*, float);
        void integrate(float, float);
        void integrateNSteps(float, float, int);
        void traverse();

    };
// Traverse function as member of class, so it can be easily called from cython
void Octree::traverse()
    {
    ::traverse(root, 0);
    }

// Constructor. Sets number of bodies, opening angle, center, creates list with leaves & builds first tree
Octree::Octree(float* p, float* v, int n, float* center, float th)
    {
    pos = p; 
    vel = v;
    N = n;
    theta2 = th;
    leaves = new Leaf*[N];
    cent[0] = center[0];
    cent[1] = center[1];
    cent[2] = center[2];
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
float Octree::getBoxSize()
    {
    float res;
    float* _pos = new float[3*N];
    for(int i = 0; i < N; i++)
        {
        _pos[3*i] = fabs(pos[4*i]);
        _pos[3*i+1] = fabs(pos[4*i+1]);
        _pos[3*i+2] = fabs(pos[4*i+2]);
        };
    res = 2 * (*std::max_element(_pos, _pos + 3*N));
    delete[] _pos;
    return res;
    }
// Creates new root cell and fills it with bodies
void Octree::buildTree()
    {
    root = new Cell (cent, getBoxSize());  
    ((Cell*)root)->insertMultiple(pos, N, leaves);
    }
// Traverses tree and calculates potential for a leaf
float Octree::leafPot(Cell* node, Leaf* leaf)
    {
    if(node == (Cell*)leaf) return 0.0f;
    
    float p = 0;
    if((node->type) || (pow((node->side),2) / dist(node->com, leaf->com) < theta2))
        return pot(leaf->com, node->com, leaf->m, node->m);
    else
        {
        for(int i = 0; i < 8; i++)
            {
            Cell* ptr = (Cell*)(node->subp[i]);
            if(ptr != NULL)
                {
                p += leafPot(ptr, leaf);
                }
            }
        }
    return p;
    }
// Calculates energy of system
float Octree::energy()
    {
    float V = 0;
    float T = 0;
    
    #pragma omp parallel for schedule(dynamic,100)
    for(int i = 0; i < N; i++)
        {
        int idx = 4*i;
        #pragma omp atomic
        V += leafPot((Cell*)root, leaves[i]);
        
        #pragma omp atomic
        T += (leaves[i]->m) * (vel[idx]*vel[idx] + vel[idx+1]*vel[idx+1] + vel[idx+2]*vel[idx+2]);
        }
    
    return 0.5f * (T - V);
    }
// Traverses tree and calculates acceleration for a leaf
void Octree::leafAccel(Cell* node, Leaf* leaf, float* a, float eps2)
    {
    float _a[3] = {0 ,0, 0};
    if((node->type) || (pow((node->side),2) / dist(node->com, leaf->com) < theta2))
        accel(leaf->com, node->com, _a, node->m, eps2);
    else
        {
        for(int i = 0; i < 8; i++)
            {
            Cell* ptr = (Cell*)(node->subp[i]);
            if(ptr != NULL)
                {
                leafAccel(ptr, leaf, _a, eps2);
                }
            }
        }
    a[0] += _a[0];
    a[1] += _a[1];
    a[2] += _a[2];
    }
// Finds acceleration for every leaf and updates pos & vel via implizit euler integration
void Octree::integrate(float dt, float eps2)
    {
    #pragma omp parallel for schedule(dynamic,100)
    for(int i = 0; i < N; i++)
        {
        int idx = 4*i;
        float a[3] = {0,0,0};
        leafAccel((Cell*)root, leaves[i], a, eps2);
        
        vel[idx] += dt * a[0];
        vel[idx+1] += dt * a[1];
        vel[idx+2] += dt * a[2];
        
        pos[idx] += dt * vel[idx];
        pos[idx+1] += dt * vel[idx+1];
        pos[idx+2] += dt * vel[idx+2];
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
    std::cout << "Lvl " << level << ", "<< "Type = "<<((Cell*)node)->type << ", m = " << ((Cell*)node)->m 
    << ", com = (" << ((Cell*)node)->com[0] << ", " << ((Cell*)node)->com[1] << ", " << ((Cell*)node)->com[2] 
    << "), midp = (" << ((Cell*)node)->midp[0] << ", " << ((Cell*)node)->midp[1] << ", " << ((Cell*)node)->midp[2] << "), Side = " << ((Cell*)node)->side << "\n";
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
    
