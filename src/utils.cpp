#include <x86intrin.h>
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
