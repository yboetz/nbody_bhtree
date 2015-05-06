#include <iostream>
#include <algorithm>


// Calculates acceleration between p1 and p2 and stores it in a
void accel(float* p1, float* p2, float* a, float m2, float eps2)
    {
    a[0] = p2[0] - p1[0];
    a[1] = p2[1] - p1[1];
//    a[2] = p2[2] - p1[2];
       
    float f = a[0]*a[0] + a[1]*a[1] + eps2;
    f = 1.0f/sqrt(f);
    f = m2 * f * f * f;
    
    a[0] *= f;
    a[1] *= f;
//    a[2] *= f;
    }

// Returns squared distance
float dist(float* p1, float* p2)
    {
    float res[2];
    res[0] = p2[0] - p1[0];
    res[1] = p2[1] - p1[1];
//    res[2] = p2[2] - p1[2];
    
    return res[0]*res[0] + res[1]*res[1];// + res[2]*res[2];    
    }
    

class Node
    {  
    public:
        static float quadIdx[4][2];
        
        bool type;                  // Type of node: 1 == Leaf, 0 == Cell
        float side, m;
        float midp[2];
        float com[2] = {0, 0};
        Node* next = NULL;
        
        Node(float*, float);
        bool contains(float*);
        short whichQuad(float*);
        
    };
    
float Node::quadIdx[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};

// Node constructor
Node::Node(float* mp, float s)
    {
//    type = 1;
    midp[0] = mp[0];
    midp[1] = mp[1];
    side = s;
    m = 0;
    }
// Returns true if position at p is contained in Node
bool Node::contains(float* p)
    {
    float d[2];
    d[0] = (midp)[0] - p[0];
    d[1] = (midp)[1] - p[1];
    float hside = side / 2.0f;
    
    if(abs(d[0]) < hside && abs(d[1]) < hside && abs(d[2]) < hside) return 1;
    else return 0;
    }
// Returns integer value of subquadrant of position p
short Node::whichQuad(float* p)
    {
    short res = 0;
    res += (midp[0] < p[0]);
    res += 2 * (midp[1] < p[1]);
    return res;    
    }
    

class Leaf: public Node
    {
    public: 
        Leaf(float*, float);
    };
// Leaf constructor. Calls baee constructor
Leaf::Leaf(float* mp, float s) 
    : Node(mp, s)
    {
    type = 1;
    }


class Cell: public Node
    {
    public:
        Node* more = NULL;                              // Pointer to first child
        Node* subp[4] = {NULL, NULL, NULL, NULL};       // Pointer to children
        
        Cell(float*, float);
        void insert(float*, int, Leaf**);
        void insertMultiple(float*, int, Leaf**);
    
    };
// Cell constructor. Calls base constructor.
Cell::Cell(float* mp, float s) 
    : Node(mp, s)
    {
    type = 0;
    }

// Recursively nserts a body into cell. If child exists, just call insertion function of child.
// If child does not exist, create leaf and insert body into leaf.
void Cell::insert(float* p, int n, Leaf** leavesptr)
    {
    short i = whichQuad(p);
    
    if(subp[i] == NULL)
        {
        float _midp[2];
        _midp[0] = midp[0] + quadIdx[i][0] * side / 4.0f;
        _midp[1] = midp[1] + quadIdx[i][1] * side / 4.0f;
        
//        Node* _more = more;                         // temporarely store ptr to children
        more = new Leaf (_midp, side / 2.0f);       // create new leaf
        subp[i] = (Node*)more;                             // append ptr to leaf in list subp
        leavesptr[n] = (Leaf*)more;                 // Append ptr to leaf to a global list of all leaves  
                
//        ((Leaf*)more)->next = _more;                // leafs nextptr points to next leaf
        ((Leaf*)more)->m = p[3];                    // Put pos and m into leaf
        ((Leaf*)more)->com[0] = p[0];
        ((Leaf*)more)->com[1] = p[1];

        }
    else if(subp[i]->type) 
        {
        Node* _more  = subp[i];
        more = new Cell (((Leaf*)_more)->midp, ((Leaf*)_more)->side);   // Create new cell in place of leaf
        subp[i] = more;
//        ((Cell*)more)->more = _more;                                    // Copy more ptr from leaf to cell
//        ((Cell*)more)->next = ((Leaf*)_more)->next;                     // Copy all other data
        ((Cell*)more)->m = ((Leaf*)_more)->m;
        ((Cell*)more)->com[0] = ((Leaf*)_more)->com[0];
        ((Cell*)more)->com[1] = ((Leaf*)_more)->com[1];
        
        short _i = ((Cell*)more)->whichQuad(((Leaf*)_more)->com); // Calculates subQuad of leaf
        ((Cell*)more)->subp[_i] = _more;
        
//        ((Leaf*)_more)->next = more;                                        // Set parameters of leaf
        ((Leaf*)_more)->midp[0] += quadIdx[_i][0] * (((Cell*)more)->side) / 4.0f;
        ((Leaf*)_more)->midp[1] += quadIdx[_i][1] * (((Cell*)more)->side) / 4.0f;
        ((Leaf*)_more)->side = (((Cell*)more)->side) / 2.0f;
        
        ((Cell*)more)->insert(p, n, leavesptr);
        // WARNING: more ptr might not be set correctly. For first version this does not matter, it is not needed for force accum.
        
        }
    else ((Cell*)subp[i])->insert(p, n, leavesptr);               
        
    float _m = p[3];
    float M = m + _m;                               // Recalc center of mass
    com[0] = (com[0] * m + p[0] * _m) / M;
    com[1] = (com[1] * m + p[1] * _m) / M;
    m = M;
    
//    std::cout << "Type = "<<leavesptr[n]->type << ", com = " << leavesptr[n]->com[0] << ", " << leavesptr[n]->com[1] << "\n"; 
    }

void Cell::insertMultiple(float* p, int N, Leaf** leavesptr)
    {
    for(int i = 0; i<N; i++)
        {
        insert(p + 4*i, i, leavesptr);
        }
    }

void traverse(Node*, int); // Define function so it can be called in class BHTree

class BHTree
    {
    public:
        Node* root;
        Leaf** leaves;
        int N;
        float theta2;
        float cent[2];

        void buildTree(float*);
        float getBoxSize(float*);
        BHTree(float*, int, float*, float);
        ~BHTree();
        void delTree(Node*);                    // helper function for deconstructor
        void leafAccel(Cell*, Leaf*, float*, float);
        void integrate(float*, float*, float, float);
        void integrateNSteps(float*, float*, float, float, int);
        void traverse();

    };

void BHTree::traverse()
    {
    ::traverse(root, 0);
    }

BHTree::BHTree(float* pos, int n, float* center, float th)
    {
    N = n;
    theta2 = th;
    leaves = new Leaf*[N];
    cent[0] = center[0]; cent[1] = center[1];
    buildTree(pos);  
    }

BHTree::~BHTree()
    {
    delTree(root);    
    delete[] leaves;
    }
    
void BHTree::delTree(Node* node)
    {
    for(int i = 0; i < 4; i++)
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
    
float BHTree::getBoxSize(float* pos)
    {
    float res;
    float* _pos = new float[3*N];
    for(int i = 0; i < N; i++)
        {
        _pos[3*i] = fabs(pos[4*i]);
        _pos[3*i+1] = fabs(pos[4*i+1]);
        _pos[3*i+2] = fabs(pos[4*i+2]);
        };
    res = 2* (*std::max_element(_pos, _pos + 3*N) + 0.0625f);
    delete[] _pos;
    return res;
    }

void BHTree::buildTree(float* p)
    {
    root = new Cell (cent, getBoxSize(p));  
    ((Cell*)root)->insertMultiple(p, N, leaves);
    }

void BHTree::leafAccel(Cell* node, Leaf* leaf, float* a, float eps2)
    {
    float _a[2] = {0 ,0};
    if((node->type == 1) || (pow((node->side),2) / dist(node->com, leaf->com) < theta2))
        accel(leaf->com, node->com, _a, node->m, eps2);
    else
        {
        for(int i = 0; i < 4; i++)
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
//    a[2] += _a[2];
    }

void BHTree::integrate(float* pos, float* vel, float dt, float eps2)
    {
    for(int i = 0; i < N; i++)
        {
        int idx = 4*i;
        float a[2] = {0,0};
        leafAccel((Cell*)root, leaves[i], a, eps2);
        
        vel[idx] += dt * a[0];
        vel[idx+1] += dt * a[1];
//        vel[idx+2] += dt * a[2];
        
        pos[idx] += dt * vel[idx];
        pos[idx+1] += dt * vel[idx+1];
//        pos[idx+2] += dt * vel[idx+2];
        }
    delTree(root);
    buildTree(pos);
    }

void BHTree::integrateNSteps(float* pos, float* vel, float dt, float eps2, int n)
    {
    for(int i = 0; i < n; i++)
        {
        integrate(pos, vel, dt, eps2);
        }
    }


void traverse(Node* node, int level)
    {
    std::cout << "Lvl " << level << ": "<< "Type = "<<((Cell*)node)->type << ", m = " << ((Cell*)node)->m << ", com = " << ((Cell*)node)->com[0] << ", " << ((Cell*)node)->com[1] << 
    ", midp = " << ((Cell*)node)->midp[0] << ", " << ((Cell*)node)->midp[1] << ", Side = " << ((Cell*)node)->side << "\n";
    if(node->type == 0)
        {
        for(int i = 0; i < 4; i++)
            {
            if(((Cell*)node)->subp[i] != NULL) traverse(((Cell*)node)->subp[i], level+1);
            }
        }
    }

void prtleaves(BHTree* root)
    {
    for(int i = 0; i < root->N; i++)
        {
        Leaf* ptr = root->leaves[i];
        std::cout << "Type = "<<ptr->type << ", com = " << ptr->com[0] << ", " << ptr->com[1] << "\n";
        }
    }
    
//int main()
//    {
//    float p[] = {1,1,0,1,-1,-1,0,1,-1,1,0,1,1,-1,0,1, 1.1, 1.1, 0, 1, -1.5, -1.5,0,1, 1, 0.1, 0, 1.5, 1.1,0.9,0,1};
////    float v[] = {1,1,0,1,-1,-1,0,1,-1,1,0,1,1,-1,0,1, 1.1, 1.1, 0, 1, -1.5, -1.5,0,1, 1, 0.1, 0, 1.5, 1.1,0.9,0,1};
//    int N = sizeof(p)/16;
//    float v[4*N] = {0};
//    float cent[2] = {0,0};
//    
//    BHTree Root = BHTree(p, N, cent, 0.25f);
//    
////    traverse(Root.root, 0);
////    prtleaves(&Root);
////    std::cout << "\n";
//    
//     Root.integrateNSteps(p, v, .00001f, .001f, 1000000);

////    traverse(Root.root, 0);
//    
////    prtleaves(&Root);
//    Root.traverse();  
//    
//    return 0;  
//    }
