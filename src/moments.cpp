#include <x86intrin.h>
#include "moments.h"

// initialize moment to zero
inline void moment_init(moment &mom)
{
    mom.xx = 0.0f;
    mom.xy = 0.0f;
    mom.xz = 0.0f;
    mom.yy = 0.0f;
    mom.yz = 0.0f;
    mom.zz = 0.0f;
}

// add quad moment from subcell to cell
inline void moment_add_sub(Cell* cell, Node* sub)
{
    __m128 pos = _mm_sub_ps(sub->com, cell->com);       // relative distance of subcell to com

    if(sub->type == 0) // if type == cell add quadrupole moment of subcell
    {
        (cell->mom).xx += (((Cell*)sub)->mom).xx;
        (cell->mom).xy += (((Cell*)sub)->mom).xy;
        (cell->mom).xz += (((Cell*)sub)->mom).xz;
        (cell->mom).yy += (((Cell*)sub)->mom).yy;
        (cell->mom).yz += (((Cell*)sub)->mom).yz;
        (cell->mom).zz += (((Cell*)sub)->mom).zz;
    }
    // use parallel-axis theorem to get final quadrupole moment
    float x, y, z, m, xx, yy, d2, tx, ty;
    x = pos[0];
    y = pos[1];
    z = pos[2];
    m = sub->com[3];
    xx = x*x;
    yy = y*y;
    d2 = xx + yy + z*z;
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
