//
//  参考: http://www001.upp.so-net.ne.jp/isaku/rand.html
//
//  xor.c : coded by isaku@pb4.so-net.ne.jp 2008/12～2009/1,  2010/10, 2012/6
//  乱数 Xorshift のＳＳＥ２を使った高速化(整数版) 
//  の一部改変
//

#pragma once
#ifndef _XOR_RAND_H_
#define _XOR_RAND_H_

#include <stdint.h>

typedef struct {
     uint32_t x, y, z, w;
} xor128_t;

static void xor128_init(xor128_t *p, uint32_t seed) {
   p->x = 123456789 - seed;
   p->y = 362436069;
   p->z = 521288629;
   p->w = 88675123;
}

static inline uint32_t xor128(xor128_t *p) {
    enum { a=11,b=8,c=19 };
    uint32_t t = p->x ^ p->x << a;
    p->x = p->y;
    p->y = p->z;
    p->z = p->w;
    return p->w ^= p->w >> c ^ t ^ t >> b;
}

#include <emmintrin.h> //SSE2

union alignas(64) xor514_t {
    uint32_t u[16];
    __m128i m[4];
};

static void xor514_init(xor514_t*p, uint32_t seed) {
   p->u[0] = seed;
   for (uint32_t i = 1; i < 16; i++)
       p->u[i] = seed = 1812433253 * (seed ^ (seed>>30)) + i;
}

static __forceinline void xor514(xor514_t *p) {
    enum { a=101,b=99,c=8 };
    __m128i s, t, x, w;
    x = p->m[0];
    w = p->m[3];
    s = _mm_slli_si128(_mm_slli_epi64(x,a-64), 8);
    t = _mm_xor_si128(x, s);
    p->m[0] = p->m[1];
    p->m[1] = p->m[2];
    p->m[2] = w; 
    s = _mm_srli_si128(_mm_srli_epi64(t,b-64), 8);
    t = _mm_xor_si128(t,s);
    s = _mm_srli_si128(w,c/8);
    s = _mm_xor_si128(s,t);
    p->m[3] = _mm_xor_si128(w,s);
}

#include <immintrin.h>

union alignas(64) xor514x2_t {
    uint32_t u[32];
    __m128i m[8];
    __m256i n[4];
};

static void xor514x2_init(xor514x2_t*p, uint32_t seed) {
   p->u[0] = seed;
   for (uint32_t i = 1; i < 32; i++)
       p->u[i] = seed = 1812433253 * (seed ^ (seed>>30)) + i;
}

static __forceinline void xor514x2(xor514x2_t *p) {
    enum { a=101,b=99,c=8 };
    __m256i s, t, x, w;
    x = p->n[0];
    w = p->n[3];
    s = _mm256_slli_si256(_mm256_slli_epi64(x,a-64), 8);
    t = _mm256_xor_si256(x, s);
    p->n[0] = p->n[1];
    p->n[1] = p->n[2];
    p->n[2] = w; 
    s = _mm256_srli_si256(_mm256_srli_epi64(t,b-64), 8);
    t = _mm256_xor_si256(t,s);
    s = _mm256_srli_si256(w,c/8);
    s = _mm256_xor_si256(s,t);
    p->n[3] = _mm256_xor_si256(w,s);
}

union alignas(64) xor514x4_t {
    uint32_t u[64];
    __m128i m[16];
    __m256i n[8];
    __m512i z[4];
};

static void xor514x4_init(xor514x4_t*p, uint32_t seed) {
    p->u[0] = seed;
    for (uint32_t i = 1; i < 32; i++)
        p->u[i] = seed = 1812433253 * (seed ^ (seed>>30)) + i;
}

static __forceinline void xor514x4(xor514x4_t *p) {
    enum { a=101, b=99, c=8 };
    __m512i s, t, x, w;
    x = p->z[0];
    w = p->z[3];
    s = _mm512_bslli_epi128(_mm512_slli_epi64(x, a-64), 8);
    t = _mm512_xor_si512(x, s);
    p->z[0] = p->z[1];
    p->z[1] = p->z[2];
    p->z[2] = w;
    s = _mm512_bsrli_epi128(_mm512_srli_epi64(t, b-64), 8);
    t = _mm512_xor_si512(t, s);
    s = _mm512_bsrli_epi128(w, c/8);
    s = _mm512_xor_si512(s, t);
    p->z[3] = _mm512_xor_si512(w, s);
}

#endif //_XOR_RAND_H_
