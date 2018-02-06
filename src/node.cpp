#include <x86intrin.h>
#include "node.h"
   
// Leaf constructor. Calls Node constructor & sets type
Leaf::Leaf(int index)
    {
    type = 1;
    id = index;
    }
// Cell constructor. Calls Node constructor & sets type
Cell::Cell(__m128 mp)
    {
    type = 0;
    midp = mp;
    }
// Returns integer value of suboctant of position p
short Cell::whichOct(__m128 p)
    {
    __m128 c = _mm_cmplt_ps(midp,p);
    c = _mm_and_ps(_mm_setr_ps(1.0f,2.0f,4.0f,0.0f),c);

    return (short)(c[0] + c[1] + c[2]);
    }
