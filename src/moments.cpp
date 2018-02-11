#include <x86intrin.h>
#include "moments.h"

// initialize moment to zero
inline void moment_init(Cell* cell)
{
    cell->com = _mm_set1_ps(0.0f);
    (cell->mom).xx = 0.0f;
    (cell->mom).xy = 0.0f;
    (cell->mom).xz = 0.0f;
    (cell->mom).yy = 0.0f;
    (cell->mom).yz = 0.0f;
    (cell->mom).zz = 0.0f;
}

// add quad. moment from subcell to cell
inline void moment_add_sub(Cell* cell, Node* sub)
{
    __m128 pos = _mm_sub_ps(sub->com, cell->com);           // relative distance of subcell to com
    // if sub->type == cell add quadrupole moment of subcell
    if(sub->type == 0)
    {
        (cell->mom).xx += (((Cell*)sub)->mom).xx;
        (cell->mom).xy += (((Cell*)sub)->mom).xy;
        (cell->mom).xz += (((Cell*)sub)->mom).xz;
        (cell->mom).yy += (((Cell*)sub)->mom).yy;
        (cell->mom).yz += (((Cell*)sub)->mom).yz;
        (cell->mom).zz += (((Cell*)sub)->mom).zz;
    }
    // use parallel-axis theorem to get final quadrupole moment
    float x, y, z, m, d2, tx, ty;
    x = pos[0];
    y = pos[1];
    z = pos[2];
    m = sub->com[3];
    d2 = x*x;
    d2 += y*y;
    d2 += z*z;
    tx = m*x;
    ty = m*y;

    (cell->mom).xy += tx*y;
    (cell->mom).xz += tx*z;
    (cell->mom).yz += ty*z;
    tx *= x;
    ty *= y;
    m *= d2 / 3.0f;
    (cell->mom).xx += tx - m;
    (cell->mom).yy += ty - m;
    (cell->mom).zz -= tx + ty - 2*m;
}
