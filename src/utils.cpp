#include "utils.h"

// Calculates cross product of vectors a & b. Last element is set to zero
inline __m128 cross_ps(__m128 a, __m128 b)
    {
    __m128 res = _mm_sub_ps(
                            _mm_mul_ps(a,_mm_permute_ps(b,_MM_SHUFFLE(3,0,2,1))),
                            _mm_mul_ps(b,_mm_permute_ps(a,_MM_SHUFFLE(3,0,2,1)))
                            );
    return _mm_permute_ps(res,_MM_SHUFFLE(3,0,2,1));
    }
// Returns distance between p1 & p2
inline float dist(__m128 p1, __m128 p2)
    {
    __m128 d = _mm_sub_ps(p2,p1);
    d = _mm_dp_ps(d,d,0b01111111);
    d = _mm_sqrt_ps(d);

    return d[0];
    }
// Returns cooked distance between center of mass & midpoint (max norm - side/2)
inline float cdist(__m128 midp, __m128 p)
    {
    __m128 res = _mm_sub_ps(p, midp);
    res = _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)),res);
    res = _mm_max_ps(res,_mm_permute_ps(res,_MM_SHUFFLE(3,1,0,2)));
    res = _mm_max_ps(res,_mm_permute_ps(res,_MM_SHUFFLE(3,1,0,2)));
    res = _mm_fmadd_ps(_mm_set1_ps(-0.5f),_mm_permute_ps(midp,0b11111111),res);

    return res[0];
    }
// compare function to sort cells according to number of bodies
inline bool compare(Node* a, Node* b)
{
    if(a->type && b->type)
        return ((Leaf*)a)->id < ((Leaf*)b)->id;
    else if(b->type)
        return false;
    else if(a->type)
        return true;
    else
        return ((Cell*)a)->n < ((Cell*)b)->n;
}
// get the interaction list for a critCell
void get_int_list(Cell* critCell, float theta, Node* start, Node* end, vector<float> &int_l, vector<float> &int_c)
{
    int_l.resize(0);
    int_c.resize(0);
    Node* node = start;
    do
    {
        Cell* _node = (Cell*)node;
        const float* begin_com = (float*)&(node->com);          // pointer to first element of center of mass
        const float* begin_mom = (float*)&(((Cell*)node)->mom); // only defined if type of node == cell
        if(node->type)
        {
            int_l.insert(int_l.end(), begin_com, begin_com + SIZEOF_COM); // append center of mass vector
            node = node->next;
        }
        // if critCell == leaf we have to calculate cdist using com instead of midp
        else if((critCell->type == 0 && ((_node->midp[3])/theta + _node->delta) < cdist(critCell->midp, _node->com)) ||
                (critCell->type == 1 && ((_node->midp[3])/theta + _node->delta) < cdist(_mm_blend_ps(critCell->com, _mm_set1_ps(0.0f), 0b1000), _node->com)))
        {
            int_c.insert(int_c.end(), begin_com, begin_com + SIZEOF_COM);   // append center of mass vector
            int_c.insert(int_c.end(), begin_mom, begin_mom + SIZEOF_MOM);   // append moment tensor struct
            node = node->next;
        }
        else node = _node->more;
    }
    while(node != end);

    // if list are not 2-aligned, add another entry
    if(int_l.size() % (2*SIZEOF_COM) != 0)
        int_l.insert(int_l.end(), SIZEOF_COM, 0.0f);
    if(int_c.size() % (2*SIZEOF_TOT) != 0)
        int_c.insert(int_c.end(), SIZEOF_TOT, 0.0f);
}
// get all leafs in critCell
void get_leaves_in_cell(Cell* critCell, vector<int> &idx, vector<float> &p, vector<float> &v, float* pos, float* vel)
{
    idx.resize(0);
    p.resize(0);
    v.resize(0);
    // append idx, pos & vel of leaves in critcell to list
    Node* node = critCell;
    Node* end = critCell->next;
    do
    {
        if(node->type)
        {
            int id = 4*((Leaf*)node)->id;
            idx.push_back(id);
            p.insert(p.end(), pos + id, pos + id + SIZEOF_COM);
            v.insert(v.end(), vel + id, vel + id + SIZEOF_COM);
            node = node->next;
        }
        else node = ((Cell*)node)->more;
    }
    while(node != end);
}
// calculate accelerations on CPU
void accel_CPU(vector<float> &pos, vector<float> &vel, vector<float> int_l, vector<float> int_c,
               float dt, float eps)
{
    __m128 dtv = _mm_setr_ps(dt, dt, dt, 0.0f);
    __m256 epsv = _mm256_set1_ps(eps);

    for(int i = 0; i < (int)pos.size(); i+=4)                       // loop over all leaves in critCell
    {
        float* cells_ptr = int_c.data();                            // pointer to first element in interaction list (better performance)
        __m128 p = _mm_load_ps(pos.data() + i);
        __m128 v = _mm_load_ps(vel.data() + i);
        __m128 a = _mm_set1_ps(0.0f);
        __m256 _p1 = _mm256_set_m128(p, p);

        for(int j = 0; j < (int)int_l.size(); j+=2*SIZEOF_COM)
        {
            __m256 _p2 = _mm256_loadu_ps(int_l.data() + j);         // read in com of 2 particles
            a = _mm_add_ps(a, accel(_p1, _p2, epsv));               // calculate acceleration
        }
        for(int j = 0; j < (int)int_c.size(); j+=2*SIZEOF_TOT)
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
        _mm_store_ps(pos.data() + i, p);
        _mm_store_ps(vel.data() + i, v);
    }
}
// calculate energy on CPU
void energy_CPU(vector<float> &pos, vector<float> &vel, vector<float> int_l, vector<float> int_c,
                float eps, float &E)
{
    float V = 0;
    float T = 0;
    __m256 epsv = _mm256_set1_ps(eps);

    for(int i = 0; i < (int)pos.size(); i+=4)                       // loop over all leaves in critCell
    {
        float* cells_ptr = int_c.data();                            // pointer to first element in interaction list (better performance)
        __m128 p = _mm_load_ps(pos.data() + i);
        __m128 v = _mm_load_ps(vel.data() + i);
        __m256 _p1 = _mm256_set_m128(p, p);
        float _V = 0.0f;                                            // is needed to cancel out floating point errors

        for(int j = 0; j < (int)int_l.size(); j+=2*SIZEOF_COM)
        {
            __m256 _p2 = _mm256_loadu_ps(int_l.data() + j);    // read in com of 2 particles
            _V += pot(_p1, _p2, epsv);                           // calculate potential
        }
        for(int j = 0; j < (int)int_c.size(); j+=2*SIZEOF_TOT)
        {
            __m256 _p2 = _mm256_loadu2_m128(cells_ptr, cells_ptr+SIZEOF_TOT);   // read in com of 2 particles
            __m256 _q1 = _mm256_loadu2_m128(cells_ptr+SIZEOF_COM, cells_ptr+SIZEOF_TOT+SIZEOF_COM);     // read in quad. tensor
            __m256 _q2 = _mm256_loadu2_m128(cells_ptr+SIZEOF_COM+2, cells_ptr+SIZEOF_TOT+SIZEOF_COM+2); // read in quad. tensor
            _V += pot(_p1, _p2, _q1, _q2, epsv);         // calculate potential
            cells_ptr += 2*SIZEOF_TOT;                      // increment pointer by 2
        }
        V += _V;
        // calculate kinetic energy
        v = _mm_dp_ps(v, v, 0b01111111);
        T += p[3] * v[0];
    }
    E += 0.5f * (T - V);
}
