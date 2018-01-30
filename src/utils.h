#include <x86intrin.h>
#ifndef UTILS_H
#define UTILS_H

// set _mm256 vector from two _m128 vectors
#define _mm256_set_m128(va, vb) _mm256_permute2f128_ps(_mm256_castps128_ps256(va),_mm256_castps128_ps256(vb),0b00100000);

// Calculates cross product of vectors a & b. Last element is set to zero
inline __m128 cross_ps(__m128 a, __m128 b);
// Returns distance between p1 & p2
inline float dist(__m128 p1, __m128 p2);
// Returns cooked distance between center of mass & midpoint (max norm - side/2)
inline float cdist(__m128 midp, __m128 p);

#endif
