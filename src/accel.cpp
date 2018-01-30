#include <x86intrin.h>
#include "accel.h"

// Calculates acceleration on p1 by two points in p2
inline __m128 accel(__m256 p1, __m256 p2, __m256 eps)
    {
    __m256 a = _mm256_sub_ps(p2,p1);
    __m256 m = _mm256_permute_ps(p2,0b11111111);

    __m256 f = _mm256_blend_ps(a,eps,0b10001000);
    f = _mm256_dp_ps(f,f,0b11111111);

    f = _mm256_mul_ps(f,_mm256_mul_ps(f,f));
    f = _mm256_rsqrt_ps(f);
    f = _mm256_mul_ps(m,f);
    a = _mm256_mul_ps(f,a);

    a = _mm256_add_ps(a,_mm256_permute2f128_ps(a,a,1));
    return _mm256_castps256_ps128(a);
    }
// Calculates potential between p1 and two points in p2
inline float pot(__m256 p1, __m256 p2)
    {
    __m256 mask = _mm256_castsi256_ps(_mm256_cmpeq_epi64(_mm256_castps_si256(p1),_mm256_castps_si256(p2)));
    mask = _mm256_and_ps(mask,_mm256_permute_ps(mask,0b01001110));

    __m256 d = _mm256_sub_ps(p2,p1);
    __m256 m = _mm256_mul_ps(_mm256_permute_ps(p1,0b11111111),_mm256_permute_ps(p2,0b11111111));

    d = _mm256_dp_ps(d,d,0b01111111);
    d = _mm256_rsqrt_ps(d);
    d = _mm256_mul_ps(m,d);

    d = _mm256_andnot_ps(mask,d);
    d = _mm256_add_ps(d,_mm256_permute2f128_ps(d,d,1));
    return -d[0];
    }