#ifndef ACCEL_H
#define ACCEL_H

#include <x86intrin.h>

// Calculates acceleration on p1 by two points in p2
inline __m128 accel(__m256 p1, __m256 p2, __m256 eps);
// Calculates acceleration on p1 by two points in p2
inline __m128 accel(__m256 p1, __m256 p2, __m256 q1, __m256 q2, __m256 eps);
// Calculates potential between p1 and two points in p2
inline float pot(__m256 p1, __m256 p2, __m256 eps);
// Calculates potential between p1 and two points in p2
inline float pot(__m256 p1, __m256 p2, __m256 q1, __m256 q2, __m256 eps);

#endif
