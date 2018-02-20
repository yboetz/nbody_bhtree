#include "accel.h"

// Calculates acceleration on p1 by two points in p2
inline __m128 accel(__m256 p1, __m256 p2, __m256 eps)
{
    // calculate normal vector and inverse distance
    __m256 r = _mm256_sub_ps(p2, p1);                       // r = p2 - p1
    __m256 m = _mm256_permute_ps(p2, 0b11111111);           // m = mass of particles in p2
    __m256 invr = _mm256_blend_ps(r, eps, 0b10001000);      // softening length in 4th position of vector r
    invr = _mm256_dp_ps(invr, invr, 0b11111111);            // invr = r.r
    invr = _mm256_rsqrt_ps(invr);                           // invr = 1/r
    r = _mm256_mul_ps(r, invr);                             // r = n = r/r
    invr = _mm256_mul_ps(invr, invr);                       // invr = 1/r^2

    // monopole contribution
    __m256 a = _mm256_mul_ps(m, invr);                      // a = m / r^2
    a = _mm256_mul_ps(a, r);                                // a = m / r^2 * n

    a = _mm256_add_ps(a, _mm256_permute2f128_ps(a, a, 1));  // add higher 128bits to lower 128bits
    return _mm256_castps256_ps128(a);
}

// Calculates acceleration on p1 by two points in p2 and quad. moments in q1 & q2
inline __m128 accel(__m256 p1, __m256 p2, __m256 q1, __m256 q2, __m256 eps)
{
    // calculate normal vector and inverse distance
    __m256 r = _mm256_sub_ps(p2, p1);                       // r = p2 - p1
    __m256 m = _mm256_permute_ps(p2, 0b11111111);           // m = mass of particles in p2
    __m256 invr = _mm256_blend_ps(r, eps, 0b10001000);      // softening length in 4th position of vector r
    invr = _mm256_dp_ps(invr, invr, 0b11111111);            // invr = r.r
    invr = _mm256_rsqrt_ps(invr);                           // invr = 1/r
    r = _mm256_mul_ps(r, invr);                             // r = n = r/r
    invr = _mm256_mul_ps(invr, invr);                       // invr = 1/r^2

    // monopole contribution
    __m256 a = _mm256_mul_ps(m, invr);                      // a = m / r^2
    a = _mm256_mul_ps(a, r);                                // a = m / r^2 * n

    // quadrupole contribution
    __m256 qr = _mm256_mul_ps(q1, _mm256_permute_ps(r, 0b00000000));                                    // qr = q11*x1, q12*x1, q13*x1
    qr = _mm256_fmadd_ps(_mm256_permute_ps(q2, 0b00111000), _mm256_permute_ps(r, 0b00101010), qr);     // qr += q13*x3, q23*x3, q33*x3
    qr = _mm256_fmadd_ps(_mm256_shuffle_ps(q1, q2, 0b00101101), _mm256_permute_ps(r, 0b00010101), qr); // qr += q12*x2, q22*x2, q23*x2

    invr = _mm256_mul_ps(invr, invr);                       // invr = 1/r^4
    qr = _mm256_mul_ps(qr, _mm256_set1_ps(3.0f));           // qr = 3 * Q.n
    a = _mm256_fmsub_ps(invr, qr, a);                       // a = 3 * Q.n / r^4 - a

    qr = _mm256_dp_ps(qr, r, 0b01111111);                   // qr = 3 * n.Q.n
    qr = _mm256_mul_ps(qr, _mm256_set1_ps(5.0f/2.0f));      // qr = 15/2 * n.Q.n
    qr = _mm256_mul_ps(qr, r);                              // qr = 15/2 * n.Q.n * n
    a = _mm256_fmsub_ps(invr, qr, a);                       // a = 15/2 * n.Q.n / r^4 * n - a

    a = _mm256_add_ps(a, _mm256_permute2f128_ps(a, a, 1));  // a = lower 128 bits + higher 128 bits
    return _mm256_castps256_ps128(a);
}

// Calculates potential between p1 and two points in p2
inline float pot(__m256 p1, __m256 p2, __m256 eps)
{
    __m256 mask = _mm256_castsi256_ps(_mm256_cmpeq_epi64(_mm256_castps_si256(p1),
                                                         _mm256_castps_si256(p2)));
    mask = _mm256_and_ps(mask, _mm256_permute_ps(mask, 0b01001110));    // mask to check if p1 == p2

    __m256 r = _mm256_sub_ps(p2, p1);                       // r = p2 - p1
    r = _mm256_blend_ps(r, eps, 0b10001000);                // softening length in 4th position of vector r
    __m256 m1 = _mm256_permute_ps(p1, 0b11111111);
    __m256 m2 = _mm256_permute_ps(p2, 0b11111111);

    __m256 invr = _mm256_dp_ps(r, r, 0b11111111);           // invr = r*r
    invr = _mm256_rsqrt_ps(invr);                           // invr = 1/r

    // monopole contribution
    __m256 V = _mm256_mul_ps(m2, invr);                     // V = m2/r
    V = _mm256_mul_ps(V, m1);                               // V = m1*m2/r

    V = _mm256_andnot_ps(mask, V);                          // filter out potential if p1 == p2
    V = _mm256_add_ps(V, _mm256_permute2f128_ps(V, V, 1));  // V = lower 128 bits + higher 128 bits
    return V[0];
}

// Calculates potential between p1 and two points in p2
inline float pot(__m256 p1, __m256 p2, __m256 q1, __m256 q2, __m256 eps)
{
    __m256 mask = _mm256_castsi256_ps(_mm256_cmpeq_epi64(_mm256_castps_si256(p1),
                                                         _mm256_castps_si256(p2)));
    mask = _mm256_and_ps(mask, _mm256_permute_ps(mask, 0b01001110));    // mask to check for equalness of p1 and p2

    __m256 r = _mm256_sub_ps(p2, p1);                       // r = p2 - p1
    r = _mm256_blend_ps(r, eps, 0b10001000);                // softening length in 4th position of vector r
    __m256 m1 = _mm256_permute_ps(p1, 0b11111111);
    __m256 m2 = _mm256_permute_ps(p2, 0b11111111);

    __m256 invr = _mm256_dp_ps(r, r, 0b11111111);           // invr = r*r
    invr = _mm256_rsqrt_ps(invr);                           // invr = 1/r

    // monopole contribution
    __m256 V = _mm256_mul_ps(m2, invr);                     // V = m2/r
    V = _mm256_mul_ps(V, m1);                               // V = m1*m2/r

    // quadrupole contribution
    r = _mm256_mul_ps(invr, r);                             // r = n = r/r
    invr = _mm256_mul_ps(invr, _mm256_mul_ps(invr, invr));  // invr = 1/r^3

    __m256 qr = _mm256_mul_ps(q1, _mm256_permute_ps(r, 0b00000000));                                    // qr = q11*x1, q12*x1, q13*x1
    qr = _mm256_fmadd_ps(_mm256_permute_ps(q2, 0b00111000), _mm256_permute_ps(r, 0b00101010), qr);     // qr += q13*x3, q23*x3, q33*x3
    qr = _mm256_fmadd_ps(_mm256_shuffle_ps(q1, q2, 0b00101101), _mm256_permute_ps(r, 0b00010101), qr); // qr += q12*x2, q22*x2, q23*x2

    qr = _mm256_dp_ps(qr, r, 0b01111111);                   // qr = n.Q.n
    qr = _mm256_mul_ps(qr, m1);                             // qr = m1 * n.Q.n
    qr = _mm256_mul_ps(qr, _mm256_set1_ps(3.0f/2.0f));      // qr = 3/2 * m2 * n.Q.n
    V = _mm256_fmadd_ps(qr, invr, V);                       // V += 3/2 * m2 * n.Q.n / r^3

    V = _mm256_andnot_ps(mask, V);                          // filter out potential if p1 == p2
    V = _mm256_add_ps(V, _mm256_permute2f128_ps(V, V, 1));  // V = lower 128 bits + higher 128 bits
    return V[0];
}
