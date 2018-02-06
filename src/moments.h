#ifndef MOMENTS_H
#define MOMENTS_H

#include <x86intrin.h>

// components of center of mass and multipole moment tensors
typedef struct moment
{
    float xx, xy, xz, yy, yz, zz;       // quadrupole moment, Q_ij = x^i x^j - 1/3 x^2 \delta^ij
} moment;

// initialize moment to zero
inline void moment_init(moment &mom);

#include "node.h"
/* add quad moment from subcell to cell.  */
inline void moment_add_sub(Cell* cell, Node* sub);

#endif
