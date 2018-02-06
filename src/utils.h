#include <x86intrin.h>
#ifndef UTILS_H
#define UTILS_H

#define _mm256_set_m128(/* __m128 */ hi, /* __m128 */ lo) _mm256_insertf128_ps(_mm256_castps128_ps256(lo), (hi), 0x1)
#define _mm256_loadu2_m128(/* float const* */ hiaddr, /* float const* */ loaddr) _mm256_set_m128(_mm_loadu_ps(hiaddr), _mm_loadu_ps(loaddr))

// Calculates cross product of vectors a & b. Last element is set to zero
inline __m128 cross_ps(__m128 a, __m128 b);
// Returns distance between p1 & p2
inline float dist(__m128 p1, __m128 p2);
// Returns cooked distance between center of mass & midpoint (max norm - side/2)
inline float cdist(__m128 midp, __m128 p);

#endif
