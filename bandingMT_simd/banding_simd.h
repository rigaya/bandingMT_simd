//  -----------------------------------------------------------------------------------------
//    バンディング低減MT SIMD by rigaya
//  -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2012-2016 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#pragma once
#ifndef _BANDING_SIMD_H_
#define _BANDING_SIMD_H_

#include <algorithm>
#include <emmintrin.h>

//実は普通にmemcpyのほうが速いかもだけど気にしない
static void __forceinline sse2_memcpy(BYTE *dst, BYTE *src, int size) {
    if (size < 64) {
        for (int i = 0; i < size; i++)
            dst[i] = src[i];
        return;
    }
    BYTE *dst_fin = dst + size;
    BYTE *dst_aligned_fin = (BYTE *)(((size_t)(dst_fin + 15) & ~15) - 64);
    __m128i x0, x1, x2, x3;
    const int start_align_diff = (int)((size_t)dst & 15);
    if (start_align_diff) {
        x0 = _mm_loadu_si128((__m128i*)src);
        _mm_storeu_si128((__m128i*)dst, x0);
        dst += 16 - start_align_diff;
        src += 16 - start_align_diff;
    }
    for ( ; dst < dst_aligned_fin; dst += 64, src += 64) {
        x0 = _mm_loadu_si128((__m128i*)(src +  0));
        x1 = _mm_loadu_si128((__m128i*)(src + 16));
        x2 = _mm_loadu_si128((__m128i*)(src + 32));
        x3 = _mm_loadu_si128((__m128i*)(src + 48));
        _mm_stream_si128((__m128i*)(dst +  0), x0);
        _mm_stream_si128((__m128i*)(dst + 16), x1);
        _mm_stream_si128((__m128i*)(dst + 32), x2);
        _mm_stream_si128((__m128i*)(dst + 48), x3);
    }
    BYTE *dst_tmp = dst_fin - 64;
    src -= (dst - dst_tmp);
    x0 = _mm_loadu_si128((__m128i*)(src +  0));
    x1 = _mm_loadu_si128((__m128i*)(src + 16));
    x2 = _mm_loadu_si128((__m128i*)(src + 32));
    x3 = _mm_loadu_si128((__m128i*)(src + 48));
    _mm_storeu_si128((__m128i*)(dst_tmp +  0), x0);
    _mm_storeu_si128((__m128i*)(dst_tmp + 16), x1);
    _mm_storeu_si128((__m128i*)(dst_tmp + 32), x2);
    _mm_storeu_si128((__m128i*)(dst_tmp + 48), x3);
}

static __forceinline int limit_1_to_8(int value) {
    int cmp_ret = (value>=8);
    return (cmp_ret<<3) + (value & (0x07 & (~(0-cmp_ret)))) + (value == 0);
}

//SSSE3のpalignrもどき
#define palignr_sse2(a,b,i) (_mm_or_si128( _mm_slli_si128(a, 16-i), _mm_srli_si128(b, i) ))

//r0 := (mask0 & 0x80) ? b0 : a0
//SSE4.1の_mm_blendv_epi8(__m128i a, __m128i b, __m128i mask) のSSE2版のようなもの
static __forceinline __m128i select_by_mask(__m128i a, __m128i b, __m128i mask) {
    return _mm_or_si128( _mm_andnot_si128(mask,a), _mm_and_si128(b,mask) );
}

//SSSE3の_mm_abs_epi16() のSSE2版
static __forceinline __m128i abs_epi16_sse2(__m128i a) {
    __m128i xMask = _mm_cmpgt_epi16(_mm_setzero_si128(), a);
    a = _mm_xor_si128(a, xMask);
    return _mm_sub_epi16(a, xMask);
}
#define xZero  (_mm_setzero_si128())
#define xOne (_mm_srli_epi16(_mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128()), 15))
#define xTwo (_mm_slli_epi16(xOne, 1))
#define xEight (_mm_slli_epi16(xOne, 3))
#define abs_epi16_simd(a) ((simd & SSSE3) ? _mm_abs_epi16((a)) : abs_epi16_sse2((a)))
#define palignr_epi8_simd(a, b, i) ((simd & SSSE3) ? _mm_alignr_epi8(a,b,i) : palignr_sse2(a,b,i))
#define blendv_epi8_simd(a, b, mask) ((simd & SSE41) ? _mm_blendv_epi8((a), (b), (mask)) : select_by_mask((a), (b), (mask)))
#define xArray128_8bit  (_mm_slli_epi64(_mm_packs_epi16(xOne, xOne), 7))
#define xArray128_16bit (_mm_slli_epi16(xOne, 7))
#define cvtlo_epi8_epi16(a) ((simd & SSE41) ? (_mm_cvtepi8_epi16((a))) : (_mm_sub_epi16(_mm_unpacklo_epi8(_mm_add_epi8((a), (xArray128_8bit)), (xZero)), (xArray128_16bit))))
#define cvthi_epi8_epi16(a) ((simd & SSE41) ? (_mm_cvtepi8_epi16(_mm_srli_si128((a), 8))) : (_mm_sub_epi16(_mm_unpackhi_epi8(_mm_add_epi8((a), (xArray128_8bit)), (xZero)), (xArray128_16bit))))
#define min3(x, y, z) (std::min((x), (std::min((y), (z)))))
#define _mm_multi6_epi32(a) (_mm_add_epi32(_mm_slli_epi32(a, 1), _mm_slli_epi32(a, 2)))

alignas(64) static const uint16_t x_range_offset[32] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};
alignas(64) static const int ref_offset[32] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};
alignas(64) static const int ref_offset_x6_plus_one[32] = {
     1,   7,  13,  19,  25,  31,  37,  43,  49,  55,  61,  67,  73,  79,  85,  91,
    97, 103, 109, 115, 121, 127, 133, 139, 145, 151, 157, 163, 169, 175, 181, 187
};

static __forceinline __m128i apply_field_mask_128(__m128i xRef, BOOL to_lower_byte, DWORD simd) {
    __m128i xFeildMask = _mm_slli_epi16(_mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128()), 1);
    if (!to_lower_byte)
        xFeildMask = palignr_epi8_simd(xFeildMask, xFeildMask, 1);
    __m128i yMaskNeg = _mm_cmpgt_epi8(_mm_setzero_si128(), xRef);
    __m128i xFieldMaskHit = _mm_andnot_si128(xFeildMask, xRef);
    xFieldMaskHit = _mm_and_si128(xFieldMaskHit, yMaskNeg);
    xRef = _mm_and_si128(xRef, xFeildMask);
    xRef = _mm_add_epi16(xRef, _mm_slli_epi16(xFieldMaskHit, 1));
    return xRef;
}

#define SWAP(type,a,b) { type temp = a; a = b; b = temp; }

#if USE_SSE
//mode012共通 ... ref用乱数の見を発生させる
static void __forceinline createRandsimd_0(char *ref_ptr, xor514_t *gen_rand, __m128i xRangeYLimit, __m128i& xRangeXLimit0, __m128i& xRangeXLimit1, const DWORD simd) {
    __m128i x0, x1;
            //本当は_mm_min_epu16の方がいいが、まあ32768を超えることもないはずなので、SSE4.1がなければ_mm_min_epi16で代用
    __m128i xRange = (simd & SSE41) ? _mm_min_epu16(xRangeYLimit, _mm_min_epu16(xRangeXLimit0, xRangeXLimit1))
                                    : _mm_min_epi16(xRangeYLimit, _mm_min_epi16(xRangeXLimit0, xRangeXLimit1));
    __m128i xRange2 = _mm_adds_epu16(_mm_slli_epi16(xRange, 1), xOne);

    xor514(gen_rand);
    x0 = _mm_unpacklo_epi8(gen_rand->m[3], _mm_setzero_si128());
    x1 = _mm_unpackhi_epi8(gen_rand->m[3], _mm_setzero_si128());
    x0 = _mm_mullo_epi16(x0, xRange2);
    x1 = _mm_mullo_epi16(x1, xRange2);
    x0 = _mm_srai_epi16(x0, 8);
    x1 = _mm_srai_epi16(x1, 8);
    x0 = _mm_or_si128(x0, _mm_slli_si128(x1, 1));
    x0 = _mm_sub_epi8(x0, _mm_or_si128(xRange, _mm_slli_si128(xRange, 1)));
    _mm_store_si128((__m128i*)(ref_ptr), x0);
    
    xRangeXLimit0 = _mm_adds_epu16(xRangeXLimit0, xEight);
    xRangeXLimit1 = _mm_subs_epu16(xRangeXLimit1, xEight);
}
//mode12用 ... dither用乱数のうち2つを発生させる、最後の一つは次で
static void __forceinline createRandsimd_1(short *dither_ptr, xor514_t *gen_rand, const short *ditherYC2, const short *ditherYC) {
    __m128i x0, x1;
    xor514(gen_rand);
    x0 = _mm_unpacklo_epi8(gen_rand->m[3], _mm_setzero_si128());
    x1 = _mm_unpackhi_epi8(gen_rand->m[3], _mm_setzero_si128());
    x0 = _mm_mullo_epi16(x0, _mm_load_si128((__m128i*)(ditherYC2 + 0)));
    x1 = _mm_mullo_epi16(x1, _mm_load_si128((__m128i*)(ditherYC2 + 8)));
    x0 = _mm_srai_epi16(x0, 8);
    x1 = _mm_srai_epi16(x1, 8);
    x0 = _mm_sub_epi16(x0, _mm_load_si128((__m128i*)(ditherYC + 0)));
    x1 = _mm_sub_epi16(x1, _mm_load_si128((__m128i*)(ditherYC + 8)));
    _mm_store_si128((__m128i*)(dither_ptr + 0), x0);
    _mm_store_si128((__m128i*)(dither_ptr + 8), x1);
}
//mode12用 ... 次で使うref用乱数とdither用乱数の最後の一つを発生させる
static void __forceinline createRandsimd_2(short *dither_ptr, char *ref_ptr, xor514_t *gen_rand, const short *ditherYC2, const short *ditherYC, __m128i xRangeYLimit, __m128i& xRangeXLimit0, __m128i& xRangeXLimit1, const DWORD simd) {
    __m128i x0, x1;
            //本当は_mm_min_epu16の方がいいが、まあ32768を超えることもないはずなので、SSE4.1がなければ_mm_min_epi16で代用
    __m128i xRange = (simd & SSE41) ? _mm_min_epu16(xRangeYLimit, _mm_min_epu16(xRangeXLimit0, xRangeXLimit1))
                                    : _mm_min_epi16(xRangeYLimit, _mm_min_epi16(xRangeXLimit0, xRangeXLimit1));
    __m128i xRange2 = _mm_adds_epu16(_mm_slli_epi16(xRange, 1), xOne);

    xor514(gen_rand);
    x0 = _mm_unpackhi_epi8(gen_rand->m[3], _mm_setzero_si128());
    x0 = _mm_mullo_epi16(x0, _mm_load_si128((__m128i*)(ditherYC2 + 16)));
    x0 = _mm_srai_epi16(x0, 8);
    x0 = _mm_sub_epi16(x0, _mm_load_si128((__m128i*)(ditherYC + 16)));
    _mm_store_si128((__m128i*)(dither_ptr + 16), x0);
    
    x0 = _mm_unpacklo_epi8(gen_rand->m[3], _mm_setzero_si128());
    x1 = _mm_unpacklo_epi8(_mm_xor_si128(gen_rand->m[3], _mm_srli_si128(gen_rand->m[3], 8)), _mm_setzero_si128());
    x0 = _mm_mullo_epi16(x0, xRange2);
    x1 = _mm_mullo_epi16(x1, xRange2);
    x0 = _mm_srai_epi16(x0, 8);
    x1 = _mm_srai_epi16(x1, 8);
    x0 = _mm_or_si128(x0, _mm_slli_si128(x1, 1));
    x0 = _mm_sub_epi8(x0, _mm_or_si128(xRange, _mm_slli_si128(xRange, 1)));
    _mm_store_si128((__m128i*)(ref_ptr), x0);
    
    xRangeXLimit0 = _mm_adds_epu16(xRangeXLimit0, xEight);
    xRangeXLimit1 = _mm_subs_epu16(xRangeXLimit1, xEight);
}
//mode0用 ... ref用乱数のみを発生させる
static void __forceinline createRandsimd_3(char *ref_ptr, xor514_t *gen_rand, __m128i xRangeYLimit, __m128i& xRangeXLimit0, __m128i& xRangeXLimit1, const DWORD simd) {
    __m128i x0, x1;
            //本当は_mm_min_epu16の方がいいが、まあ32768を超えることもないはずなので、SSE4.1がなければ_mm_min_epi16で代用
    __m128i xRange = (simd & SSE41) ? _mm_min_epu16(xRangeYLimit, _mm_min_epu16(xRangeXLimit0, xRangeXLimit1))
                                    : _mm_min_epi16(xRangeYLimit, _mm_min_epi16(xRangeXLimit0, xRangeXLimit1));
    __m128i xRange2 = _mm_adds_epu16(_mm_slli_epi16(xRange, 1), xOne);

    xor514(gen_rand);
    x0 = _mm_unpacklo_epi8(gen_rand->m[3], _mm_setzero_si128());
    x1 = _mm_unpackhi_epi8(gen_rand->m[3], _mm_setzero_si128());
    x0 = _mm_mullo_epi16(x0, xRange2);
    x1 = _mm_mullo_epi16(x1, xRange2);
    x0 = _mm_srai_epi16(x0, 8);
    x1 = _mm_srai_epi16(x1, 8);
    x0 = _mm_or_si128(x0, _mm_slli_si128(x1, 1));
    x0 = _mm_sub_epi8(x0, _mm_or_si128(xRange, _mm_slli_si128(xRange, 1)));
    _mm_store_si128((__m128i*)(ref_ptr), x0);
    
    xRangeXLimit0 = _mm_adds_epu16(xRangeXLimit0, xEight);
    xRangeXLimit1 = _mm_subs_epu16(xRangeXLimit1, xEight);
}
static void __forceinline createRandsimd_4(xor514_t *gen_rand, const short *ditherYC2, const short *ditherYC, __m128i& x0, __m128i& x1, __m128i& x2) {
    xor514(gen_rand);
    x0 = _mm_unpacklo_epi8(gen_rand->m[3], _mm_setzero_si128());
    x1 = _mm_unpackhi_epi8(gen_rand->m[3], _mm_setzero_si128());
    x2 = _mm_unpacklo_epi8(_mm_xor_si128(gen_rand->m[3], _mm_srli_si128(gen_rand->m[3], 8)), _mm_setzero_si128());
    x0 = _mm_mullo_epi16(x0, _mm_load_si128((__m128i*)(ditherYC2 +  0)));
    x1 = _mm_mullo_epi16(x1, _mm_load_si128((__m128i*)(ditherYC2 +  8)));
    x2 = _mm_mullo_epi16(x2, _mm_load_si128((__m128i*)(ditherYC2 + 16)));
    x0 = _mm_srai_epi16(x0, 8);
    x1 = _mm_srai_epi16(x1, 8);
    x2 = _mm_srai_epi16(x2, 8);
    x0 = _mm_sub_epi16(x0, _mm_load_si128((__m128i*)(ditherYC +  0)));
    x1 = _mm_sub_epi16(x1, _mm_load_si128((__m128i*)(ditherYC +  8)));
    x2 = _mm_sub_epi16(x2, _mm_load_si128((__m128i*)(ditherYC + 16)));
}


//process_per_field、simdは定数として与え、
//条件分岐をコンパイル時に削除させる
static void __forceinline decrease_banding_mode0_simd(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip, BOOL process_per_field, DWORD simd) {
    const int sample_mode = 0;
    const int max_w  = fpip->max_w;
    const int width = fpip->w;
    const int height = fpip->h;
    const int seed    = fp->track[7];
    const int ditherY = fp->track[4];
    const int ditherC = fp->track[5];
    const int rand_each_frame = fp->check[1];
    const int blur_first      = fp->check[0];
    const int range           = fp->track[0];
    const int threshold_y     = fp->track[1] << (!(sample_mode && blur_first) + 1);
    const int threshold_cb    = fp->track[2] << (!(sample_mode && blur_first) + 1);
    const int threshold_cr    = fp->track[3] << (!(sample_mode && blur_first) + 1);
    const int b_start = (band.block_count_x * band.block_count_y *  thread_id) / thread_num;
    const int b_end   = (band.block_count_x * band.block_count_y * (thread_id+1)) / thread_num;
    __m128i xRefMulti  = _mm_unpacklo_epi16(_mm_set1_epi16(max_w), xOne);
    
    alignas(16) char     ref[16];
    alignas(16) PIXEL_YC ycp_buffer[8];
    alignas(16) int      ref_buffer[8];
    alignas(16) short    threshold[24];

    for (int i = 0; i < 8; i++) {
        threshold[3*i+0] = threshold_y;
        threshold[3*i+1] = threshold_cb;
        threshold[3*i+2] = threshold_cr;
    }
    xor514_t gen_rand;
    if (!rand_each_frame) {
        xor514_init(&gen_rand, seed + (b_start << 4));
    } else {
        gen_rand = band.gen_rand[thread_id];
    }

    for (int ib = b_start; ib < b_end; ib++) {
        int x_start, x_end, y_start, y_end;
        band_get_block_range(ib, width, height, &x_start, &x_end, &y_start, &y_end);
        int y;
        const int y_main_end = std::min(y_end, height - 1);
        for (y = y_start; y < y_main_end; y++) {
            PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w + x_start;
            PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w + x_start;
            __m128i xRangeYLimit = _mm_set1_epi16(min3(range, y, height - y - 1));
            __m128i xRangeXLimit0 = _mm_add_epi16(_mm_load_si128((__m128i*)x_range_offset), _mm_set1_epi16(x_start));
            __m128i xRangeXLimit1 = _mm_subs_epu16(_mm_set1_epi16(width - x_start - 1), _mm_load_si128((__m128i*)x_range_offset));
            createRandsimd_0(ref, &gen_rand, xRangeYLimit, xRangeXLimit0, xRangeXLimit1, simd);
            for (int i_step = 0, x = (x_end - x_start) - 8; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {
                __m128i xRef = _mm_loadu_si128((__m128i*)ref);
                if (process_per_field)
                    xRef = apply_field_mask_128(xRef, TRUE, simd);
                __m128i xRefLower = cvtlo_epi8_epi16(xRef);
                __m128i xRefUpper = cvthi_epi8_epi16(xRef);

                xRefLower = _mm_add_epi32(_mm_madd_epi16(xRefLower, xRefMulti), _mm_load_si128((__m128i*)&ref_offset[0]));
                xRefUpper = _mm_add_epi32(_mm_madd_epi16(xRefUpper, xRefMulti), _mm_load_si128((__m128i*)&ref_offset[4]));

                _mm_store_si128((__m128i*)&ref_buffer[0], _mm_multi6_epi32(xRefLower));
                _mm_store_si128((__m128i*)&ref_buffer[4], _mm_multi6_epi32(xRefUpper));

                if (simd & SSSE3) {

                    __m64 m0, m1, m2, m3, m4, m5, m6, m7;
                    m1 = *(__m64*)((BYTE *)ycp_src + ref_buffer[1]);
                    m2 = *(__m64*)((BYTE *)ycp_src + ref_buffer[2]);
                    m3 = *(__m64*)((BYTE *)ycp_src + ref_buffer[3]);
                    m5 = *(__m64*)((BYTE *)ycp_src + ref_buffer[5]);
                    m6 = *(__m64*)((BYTE *)ycp_src + ref_buffer[6]);
                    m7 = *(__m64*)((BYTE *)ycp_src + ref_buffer[7]);
                    m0 = m2;
                    m4 = m6;
                    m0 = _mm_slli_si64(m0, 16);
                    m4 = _mm_slli_si64(m4, 16);
                    m3 = _mm_alignr_pi8(m3, m0, 6);
                    m7 = _mm_alignr_pi8(m7, m4, 6);
                    *(__m64 *)((BYTE *)ycp_buffer + 16) = m3;
                    *(__m64 *)((BYTE *)ycp_buffer + 40) = m7;
                    m0 = m1;
                    m4 = m5;
                    m0 = _mm_slli_si64(m0, 16);
                    m4 = _mm_slli_si64(m4, 16);
                    m2 = _mm_alignr_pi8(m2, m0, 4);
                    m6 = _mm_alignr_pi8(m6, m4, 4);
                    m3 = *(__m64*)((BYTE *)ycp_src + ref_buffer[0]);
                    m7 = *(__m64*)((BYTE *)ycp_src + ref_buffer[4]);
                    *(__m64 *)((BYTE *)ycp_buffer +  8) = m2;
                    *(__m64 *)((BYTE *)ycp_buffer + 32) = m6;
                    m3 = _mm_slli_si64(m3, 16);
                    m7 = _mm_slli_si64(m7, 16);
                    m1 = _mm_alignr_pi8(m1, m3, 2);
                    m5 = _mm_alignr_pi8(m5, m7, 2);
                    *(__m64 *)((BYTE *)ycp_buffer +  0) = m1;
                    *(__m64 *)((BYTE *)ycp_buffer + 24) = m5;

                } else {

                    ycp_buffer[0] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[0]);
                    ycp_buffer[1] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[1]);
                    ycp_buffer[2] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[2]);
                    ycp_buffer[3] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[3]);
                    ycp_buffer[4] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[4]);
                    ycp_buffer[5] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[5]);
                    ycp_buffer[6] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[6]);
                    ycp_buffer[7] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[7]);

                }
                createRandsimd_3(ref, &gen_rand, xRangeYLimit, xRangeXLimit0, xRangeXLimit1, simd);

                __m128i xYCPRef0, xYCPDiff, xYCP, xThreshold, xBase, xMask;

                xYCPRef0 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +   0));
                xYCP     = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src +  0));
                xYCPDiff = abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef0));
                xThreshold = _mm_load_si128((__m128i*)(threshold +  0));
                xMask    = _mm_cmpgt_epi16(xThreshold, xYCPDiff); //(a > b) ? 0xffff : 0x0000;
                xBase    = blendv_epi8_simd(xYCP, xYCPRef0, xMask);  //r = (mask0 & 0xff) ? b : a
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst +  0), xBase);


                xYCPRef0 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +  16));
                xYCP     = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src + 16));
                xYCPDiff = abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef0));
                xThreshold = _mm_load_si128((__m128i*)(threshold +  8));
                xMask    = _mm_cmpgt_epi16(xThreshold, xYCPDiff); //(a > b) ? 0xffff : 0x0000;
                xBase    = blendv_epi8_simd(xYCP, xYCPRef0, xMask);  //r = (mask0 & 0xff) ? b : a
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst + 16), xBase);


                xYCPRef0 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer + 32));
                xYCP     = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src + 32));
                xYCPDiff = abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef0));
                xThreshold = _mm_load_si128((__m128i*)(threshold + 16));
                xMask    = _mm_cmpgt_epi16(xThreshold, xYCPDiff); //(a > b) ? 0xffff : 0x0000;
                xBase    = blendv_epi8_simd(xYCP, xYCPRef0, xMask);  //r = (mask0 & 0xff) ? b : a
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst + 32), xBase);

                i_step = limit_1_to_8(x);
            }
        }
        //最後のライン
        if (y < y_end)
            sse2_memcpy((BYTE *)(fpip->ycp_temp + y * max_w + x_start), (BYTE *)(fpip->ycp_edit + y * max_w + x_start), (x_end - x_start) * 6);
    }
    _mm_empty();
    band.gen_rand[thread_id] = gen_rand;
}

#pragma warning (push)
#pragma warning (disable: 4799) //warning C4799: emms命令がありません
static void __forceinline gather1(PIXEL_YC *ycp_buffer, const PIXEL_YC *ycp_src, const int *ref_buffer, DWORD simd) {            
    if (simd & SSSE3) {

        __m64 m0, m1, m2, m3, m4, m5, m6, m7;
        m1 = *(__m64*)((BYTE *)ycp_src + ref_buffer[1]);
        m2 = *(__m64*)((BYTE *)ycp_src + ref_buffer[2]);
        m3 = *(__m64*)((BYTE *)ycp_src + ref_buffer[3]);
        m5 = *(__m64*)((BYTE *)ycp_src - ref_buffer[1] + 12);
        m6 = *(__m64*)((BYTE *)ycp_src - ref_buffer[2] + 24);
        m7 = *(__m64*)((BYTE *)ycp_src - ref_buffer[3] + 36);
        m0 = m2;
        m4 = m6;
        m0 = _mm_slli_si64(m0, 16);
        m4 = _mm_slli_si64(m4, 16);
        m3 = _mm_alignr_pi8(m3, m0, 6);
        m7 = _mm_alignr_pi8(m7, m4, 6);
        *(__m64 *)((BYTE *)ycp_buffer + 16) = m3;
        *(__m64 *)((BYTE *)ycp_buffer + 64) = m7;
        m0 = m1;
        m4 = m5;
        m0 = _mm_slli_si64(m0, 16);
        m4 = _mm_slli_si64(m4, 16);
        m2 = _mm_alignr_pi8(m2, m0, 4);
        m6 = _mm_alignr_pi8(m6, m4, 4);
        m3 = *(__m64*)((BYTE *)ycp_src + ref_buffer[0]);
        m7 = *(__m64*)((BYTE *)ycp_src - ref_buffer[0] + 0);
        *(__m64 *)((BYTE *)ycp_buffer +  8) = m2;
        *(__m64 *)((BYTE *)ycp_buffer + 56) = m6;
        m3 = _mm_slli_si64(m3, 16);
        m7 = _mm_slli_si64(m7, 16);
        m1 = _mm_alignr_pi8(m1, m3, 2);
        m5 = _mm_alignr_pi8(m5, m7, 2);
        *(__m64 *)((BYTE *)ycp_buffer +  0) = m1;
        *(__m64 *)((BYTE *)ycp_buffer + 48) = m5;
    } else {
        ycp_buffer[ 0] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[0]);
        ycp_buffer[ 1] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[1]);
        ycp_buffer[ 2] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[2]);
        ycp_buffer[ 3] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[3]);
        ycp_buffer[ 8] = *(PIXEL_YC *)((BYTE *)ycp_src - ref_buffer[0]);
        ycp_buffer[ 9] = *(PIXEL_YC *)((BYTE *)ycp_src - ref_buffer[1] + 12);
        ycp_buffer[10] = *(PIXEL_YC *)((BYTE *)ycp_src - ref_buffer[2] + 24);
        ycp_buffer[11] = *(PIXEL_YC *)((BYTE *)ycp_src - ref_buffer[3] + 36);
    }
}

static void __forceinline gather2(PIXEL_YC *ycp_buffer, const PIXEL_YC *ycp_src, const int *ref_buffer, DWORD simd) {            
    if (simd & SSSE3) {

        __m64 m0, m1, m2, m3, m4, m5, m6, m7;
        m1 = *(__m64*)((BYTE *)ycp_src + ref_buffer[5]);
        m2 = *(__m64*)((BYTE *)ycp_src + ref_buffer[6]);
        m3 = *(__m64*)((BYTE *)ycp_src + ref_buffer[7]);
        m5 = *(__m64*)((BYTE *)ycp_src - ref_buffer[5] + 60);
        m6 = *(__m64*)((BYTE *)ycp_src - ref_buffer[6] + 72);
        m7 = *(__m64*)((BYTE *)ycp_src - ref_buffer[7] + 84);
        m0 = m2;
        m4 = m6;
        m0 = _mm_slli_si64(m0, 16);
        m4 = _mm_slli_si64(m4, 16);
        m3 = _mm_alignr_pi8(m3, m0, 6);
        m7 = _mm_alignr_pi8(m7, m4, 6);
        *(__m64 *)((BYTE *)ycp_buffer + 40) = m3;
        *(__m64 *)((BYTE *)ycp_buffer + 88) = m7;
        m0 = m1;
        m4 = m5;
        m0 = _mm_slli_si64(m0, 16);
        m4 = _mm_slli_si64(m4, 16);
        m2 = _mm_alignr_pi8(m2, m0, 4);
        m6 = _mm_alignr_pi8(m6, m4, 4);
        m3 = *(__m64*)((BYTE *)ycp_src + ref_buffer[4]);
        m7 = *(__m64*)((BYTE *)ycp_src - ref_buffer[4] + 48);
        *(__m64 *)((BYTE *)ycp_buffer + 32) = m2;
        *(__m64 *)((BYTE *)ycp_buffer + 80) = m6;
        m3 = _mm_slli_si64(m3, 16);
        m7 = _mm_slli_si64(m7, 16);
        m1 = _mm_alignr_pi8(m1, m3, 2);
        m5 = _mm_alignr_pi8(m5, m7, 2);
        *(__m64 *)((BYTE *)ycp_buffer + 24) = m1;
        *(__m64 *)((BYTE *)ycp_buffer + 72) = m5;

    } else {
                
        ycp_buffer[ 4] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[4]);
        ycp_buffer[ 5] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[5]);
        ycp_buffer[ 6] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[6]);
        ycp_buffer[ 7] = *(PIXEL_YC *)((BYTE *)ycp_src + ref_buffer[7]);
        ycp_buffer[12] = *(PIXEL_YC *)((BYTE *)ycp_src - ref_buffer[4] + 48);
        ycp_buffer[13] = *(PIXEL_YC *)((BYTE *)ycp_src - ref_buffer[5] + 60);
        ycp_buffer[14] = *(PIXEL_YC *)((BYTE *)ycp_src - ref_buffer[6] + 72);
        ycp_buffer[15] = *(PIXEL_YC *)((BYTE *)ycp_src - ref_buffer[7] + 84);

    }
}
#pragma warning (pop)

//blur_first、process_per_field、simdは定数として与え、
//条件分岐をコンパイル時に削除させる
static void __forceinline decrease_banding_mode1_simd(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip, BOOL blur_first,  BOOL process_per_field, DWORD simd) {
    const int sample_mode = 1;
    const int max_w = fpip->max_w;
    const int width = fpip->w;
    const int height = fpip->h;
    const int seed    = fp->track[7];
    const int ditherY = fp->track[4];
    const int ditherC = fp->track[5];
    const int rand_each_frame = fp->check[1];
    const int range           = fp->track[0];
    const int threshold_y     = fp->track[1] << (!(sample_mode && blur_first) + 1);
    const int threshold_cb    = fp->track[2] << (!(sample_mode && blur_first) + 1);
    const int threshold_cr    = fp->track[3] << (!(sample_mode && blur_first) + 1);
    const int b_start = (band.block_count_x * band.block_count_y *  thread_id) / thread_num;
    const int b_end   = (band.block_count_x * band.block_count_y * (thread_id+1)) / thread_num;
    __m128i xRefMulti  = _mm_unpacklo_epi16(_mm_set1_epi16(max_w), xOne);
    
    alignas(16) char     ref[16];
    alignas(16) short    dither[24];
    alignas(16) short    ditherYC[24];
    alignas(16) short    ditherYC2[24];
    alignas(16) PIXEL_YC ycp_buffer[16];
    alignas(16) int      ref_buffer[8];
    alignas(16) short    threshold[24];

    for (int i = 0; i < 8; i++) {
        threshold[3*i+0] = threshold_y;
        threshold[3*i+1] = threshold_cb;
        threshold[3*i+2] = threshold_cr;
    }
    for (int i = 0; i < 8; i++) {
        ditherYC[3*i+0] = ditherY;
        ditherYC[3*i+1] = ditherC;
        ditherYC[3*i+2] = ditherC;
    }
    {
        __m128i x0 = _mm_load_si128((__m128i *)(ditherYC +  0));
        __m128i x1 = _mm_load_si128((__m128i *)(ditherYC +  8));
        __m128i x2 = _mm_load_si128((__m128i *)(ditherYC + 16));
        x0 = _mm_slli_epi16(x0, 1);
        x1 = _mm_slli_epi16(x1, 1);
        x2 = _mm_slli_epi16(x2, 1);
        x0 = _mm_add_epi16(x0, xOne);
        x1 = _mm_add_epi16(x1, xOne);
        x2 = _mm_add_epi16(x2, xOne);
        _mm_store_si128((__m128i *)(ditherYC2 +  0), x0);
        _mm_store_si128((__m128i *)(ditherYC2 +  8), x1);
        _mm_store_si128((__m128i *)(ditherYC2 + 16), x2);
    }
    xor514_t gen_rand;
    if (!rand_each_frame) {
        xor514_init(&gen_rand, seed + (b_start << 4));
    } else {
        gen_rand = band.gen_rand[thread_id];
    }

    for (int ib = b_start; ib < b_end; ib++) {
        int x_start, x_end, y_start, y_end;
        band_get_block_range(ib, width, height, &x_start, &x_end, &y_start, &y_end);
        int y;
        const int y_main_end = std::min(y_end, height - 1);
        for (y = y_start; y < y_main_end; y++) {
            PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w + x_start;
            PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w + x_start;
            __m128i xRangeYLimit = _mm_set1_epi16(min3(range, y, height - y - 1));
            __m128i xRangeXLimit0 = _mm_add_epi16(_mm_load_si128((__m128i*)x_range_offset), _mm_set1_epi16(x_start));
            __m128i xRangeXLimit1 = _mm_subs_epu16(_mm_set1_epi16(width - x_start - 1), _mm_load_si128((__m128i*)x_range_offset));
            createRandsimd_0(ref, &gen_rand, xRangeYLimit, xRangeXLimit0, xRangeXLimit1, simd);
            for (int i_step = 0, x = (x_end - x_start) - 8; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {
                __m128i xRef = _mm_loadu_si128((__m128i*)ref);
                if (process_per_field)
                    xRef = apply_field_mask_128(xRef, TRUE, simd);
                __m128i xRefLower = cvtlo_epi8_epi16(xRef);
                __m128i xRefUpper = cvthi_epi8_epi16(xRef);

                xRefLower = _mm_add_epi32(_mm_madd_epi16(xRefLower, xRefMulti), _mm_load_si128((__m128i*)&ref_offset[0]));
                xRefUpper = _mm_add_epi32(_mm_madd_epi16(xRefUpper, xRefMulti), _mm_load_si128((__m128i*)&ref_offset[4]));

                _mm_store_si128((__m128i*)&ref_buffer[0], _mm_multi6_epi32(xRefLower));
                _mm_store_si128((__m128i*)&ref_buffer[4], _mm_multi6_epi32(xRefUpper));


                gather1(ycp_buffer, ycp_src, ref_buffer, simd);
                createRandsimd_1(dither, &gen_rand, ditherYC2, ditherYC);

                __m128i xYCPRef0, xYCPRef1, xYCPAvg, xYCPDiff, xYCP, xThreshold, xBase, xMask, xDither;

                xYCPRef0 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +   0));
                xYCPRef1 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +  48));
                xYCPAvg  = _mm_srai_epi16(_mm_adds_epi16(_mm_adds_epi16(xYCPRef0, xYCPRef1), xOne), 1);
                xYCP     = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src +  0));
                xYCPDiff = (blur_first) ? abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPAvg))
                    : _mm_max_epi16(abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef0)),
                        abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef1)));
                xThreshold = _mm_load_si128((__m128i*)(threshold +  0));
                xMask    = _mm_cmpgt_epi16(xThreshold, xYCPDiff); //(a > b) ? 0xffff : 0x0000;
                xBase    = blendv_epi8_simd(xYCP, xYCPAvg, xMask);  //r = (mask0 & 0xff) ? b : a
                xDither  = _mm_load_si128((__m128i *)(dither +  0));
                xYCP     = _mm_adds_epi16(xBase, xDither);
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst +  0), xYCP);


                gather2(ycp_buffer, ycp_src, ref_buffer, simd);
                createRandsimd_2(dither, ref, &gen_rand, ditherYC2, ditherYC, xRangeYLimit, xRangeXLimit0, xRangeXLimit1, simd);

                xYCPRef0 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +  16));
                xYCPRef1 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +  64));
                xYCPAvg  = _mm_srai_epi16(_mm_adds_epi16(_mm_adds_epi16(xYCPRef0, xYCPRef1), xOne), 1);
                xYCP     = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src + 16));
                xYCPDiff = (blur_first) ? abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPAvg))
                    : _mm_max_epi16(abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef0)),
                        abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef1)));
                xThreshold = _mm_load_si128((__m128i*)(threshold +  8));
                xMask    = _mm_cmpgt_epi16(xThreshold, xYCPDiff); //(a > b) ? 0xffff : 0x0000;
                xBase    = blendv_epi8_simd(xYCP, xYCPAvg, xMask);  //r = (mask0 & 0xff) ? b : a
                xDither  = _mm_load_si128((__m128i *)(dither +  8));
                xYCP     = _mm_adds_epi16(xBase, xDither);
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst + 16), xYCP);



                xYCPRef0 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +  32));
                xYCPRef1 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +  80));
                xYCPAvg  = _mm_srai_epi16(_mm_adds_epi16(_mm_adds_epi16(xYCPRef0, xYCPRef1), xOne), 1);
                xYCP     = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src + 32));
                xYCPDiff = (blur_first) ? abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPAvg))
                    : _mm_max_epi16(abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef0)),
                        abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef1)));
                xThreshold = _mm_load_si128((__m128i*)(threshold + 16));
                xMask    = _mm_cmpgt_epi16(xThreshold, xYCPDiff); //(a > b) ? 0xffff : 0x0000;
                xBase    = blendv_epi8_simd(xYCP, xYCPAvg, xMask);  //r = (mask0 & 0xff) ? b : a
                xDither  = _mm_load_si128((__m128i *)(dither + 16));
                xYCP     = _mm_adds_epi16(xBase, xDither);
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst + 32), xYCP);

                i_step = limit_1_to_8(x);
            }
        }
        //最後のライン
        if (y < y_end) {
            PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w + x_start;
            PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w + x_start;
            for (int i_step = 0, x = (x_end - x_start) - 8; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {

                __m128i xYCP0, xYCP1, xYCP2, xDither0, xDither1, xDither2;
                createRandsimd_4(&gen_rand, ditherYC2, ditherYC, xDither0, xDither1, xDither2);

                xYCP0    = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src +  0));
                xYCP1    = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src + 16));
                xYCP2    = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src + 32));
                xYCP0    = _mm_adds_epi16(xYCP0, xDither0);
                xYCP1    = _mm_adds_epi16(xYCP1, xDither1);
                xYCP2    = _mm_adds_epi16(xYCP2, xDither2);
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst +  0), xYCP0);
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst + 16), xYCP1);
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst + 32), xYCP2);

                i_step = limit_1_to_8(x);
            }
        }
    }
    _mm_empty();
    band.gen_rand[thread_id] = gen_rand;
}

//blur_first、process_per_field、simdは定数として与え、
//条件分岐をコンパイル時に削除させる
static void __forceinline decrease_banding_mode2_simd(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip, BOOL blur_first,  BOOL process_per_field, DWORD simd) {
    const int sample_mode = 2;
    const int max_w  = fpip->max_w;
    const int width = fpip->w;
    const int height = fpip->h;
    const int seed    = fp->track[7];
    const int ditherY = fp->track[4];
    const int ditherC = fp->track[5];
    const int rand_each_frame = fp->check[1];
    const int range           = fp->track[0];
    const int threshold_y     = fp->track[1] << (!(sample_mode && blur_first) + 1);
    const int threshold_cb    = fp->track[2] << (!(sample_mode && blur_first) + 1);
    const int threshold_cr    = fp->track[3] << (!(sample_mode && blur_first) + 1);
    const int b_start = (band.block_count_x * band.block_count_y *  thread_id) / thread_num;
    const int b_end   = (band.block_count_x * band.block_count_y * (thread_id+1)) / thread_num;
    __m128i xRefMulti  = _mm_unpacklo_epi16(_mm_set1_epi16(max_w), xOne);
    __m128i xRefMulti2 = _mm_unpacklo_epi16(xOne, _mm_set1_epi16(-max_w));
    
    alignas(16) char     ref[16];
    alignas(16) short    dither[24];
    alignas(16) short    ditherYC[24];
    alignas(16) short    ditherYC2[24];
    alignas(16) PIXEL_YC ycp_buffer[32];
    alignas(16) int      ref_buffer[16];
    alignas(16) short    threshold[24];

    for (int i = 0; i < 8; i++) {
        threshold[3*i+0] = threshold_y;
        threshold[3*i+1] = threshold_cb;
        threshold[3*i+2] = threshold_cr;
    }
    for (int i = 0; i < 8; i++) {
        ditherYC[3*i+0] = ditherY;
        ditherYC[3*i+1] = ditherC;
        ditherYC[3*i+2] = ditherC;
    }
    {
        __m128i x0 = _mm_load_si128((__m128i *)(ditherYC +  0));
        __m128i x1 = _mm_load_si128((__m128i *)(ditherYC +  8));
        __m128i x2 = _mm_load_si128((__m128i *)(ditherYC + 16));
        x0 = _mm_slli_epi16(x0, 1);
        x1 = _mm_slli_epi16(x1, 1);
        x2 = _mm_slli_epi16(x2, 1);
        x0 = _mm_add_epi16(x0, xOne);
        x1 = _mm_add_epi16(x1, xOne);
        x2 = _mm_add_epi16(x2, xOne);
        _mm_store_si128((__m128i *)(ditherYC2 +  0), x0);
        _mm_store_si128((__m128i *)(ditherYC2 +  8), x1);
        _mm_store_si128((__m128i *)(ditherYC2 + 16), x2);
    }
    xor514_t gen_rand;
    if (!rand_each_frame) {
        xor514_init(&gen_rand, seed + (b_start << 4));
    } else {
        gen_rand = band.gen_rand[thread_id];
    }

    for (int ib = b_start; ib < b_end; ib++) {
        int x_start, x_end, y_start, y_end;
        band_get_block_range(ib, width, height, &x_start, &x_end, &y_start, &y_end);
        int y;
        const int y_main_end = std::min(y_end, height - 1);
        for (y = y_start; y < y_main_end; y++) {
            PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w + x_start;
            PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w + x_start;
            __m128i xRangeYLimit = _mm_set1_epi16(min3(range, y, height - y - 1));
            __m128i xRangeXLimit0 = _mm_add_epi16(_mm_load_si128((__m128i*)x_range_offset), _mm_set1_epi16(x_start));
            __m128i xRangeXLimit1 = _mm_subs_epu16(_mm_set1_epi16(width - x_start - 1), _mm_load_si128((__m128i*)x_range_offset));
            createRandsimd_0(ref, &gen_rand, xRangeYLimit, xRangeXLimit0, xRangeXLimit1, simd);
            for (int i_step = 0, x = (x_end - x_start) - 8; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {
                __m128i xRef = _mm_loadu_si128((__m128i*)ref);
                if (process_per_field) {
                    __m128i xRef2 = apply_field_mask_128(xRef, TRUE, simd);
                    __m128i xRefLower = cvtlo_epi8_epi16(xRef2);
                    __m128i xRefUpper = cvthi_epi8_epi16(xRef2);

                    _mm_store_si128((__m128i*)&ref_buffer[0], _mm_multi6_epi32(_mm_add_epi32(_mm_madd_epi16(xRefLower, xRefMulti), _mm_load_si128((__m128i*)&ref_offset[0]))));
                    _mm_store_si128((__m128i*)&ref_buffer[4], _mm_multi6_epi32(_mm_add_epi32(_mm_madd_epi16(xRefUpper, xRefMulti), _mm_load_si128((__m128i*)&ref_offset[4]))));

                    xRef2 = apply_field_mask_128(xRef, FALSE, simd);
                    xRefLower = cvtlo_epi8_epi16(xRef2);
                    xRefUpper = cvthi_epi8_epi16(xRef2);

                    _mm_store_si128((__m128i*)&ref_buffer[8], _mm_multi6_epi32(_mm_add_epi32(_mm_madd_epi16(xRefLower, xRefMulti2), _mm_load_si128((__m128i*)&ref_offset[0]))));
                    _mm_store_si128((__m128i*)&ref_buffer[12], _mm_multi6_epi32(_mm_add_epi32(_mm_madd_epi16(xRefUpper, xRefMulti2), _mm_load_si128((__m128i*)&ref_offset[4]))));
                } else {
                    __m128i xRefLower = cvtlo_epi8_epi16(xRef);
                    __m128i xRefUpper = cvthi_epi8_epi16(xRef);

                    _mm_store_si128((__m128i*)&ref_buffer[0], _mm_multi6_epi32(_mm_add_epi32(_mm_madd_epi16(xRefLower, xRefMulti), _mm_load_si128((__m128i*)&ref_offset[0]))));
                    _mm_store_si128((__m128i*)&ref_buffer[4], _mm_multi6_epi32(_mm_add_epi32(_mm_madd_epi16(xRefUpper, xRefMulti), _mm_load_si128((__m128i*)&ref_offset[4]))));

                    _mm_store_si128((__m128i*)&ref_buffer[8], _mm_multi6_epi32(_mm_add_epi32(_mm_madd_epi16(xRefLower, xRefMulti2), _mm_load_si128((__m128i*)&ref_offset[0]))));
                    _mm_store_si128((__m128i*)&ref_buffer[12], _mm_multi6_epi32(_mm_add_epi32(_mm_madd_epi16(xRefUpper, xRefMulti2), _mm_load_si128((__m128i*)&ref_offset[4]))));
                }

                gather1(ycp_buffer +  0, ycp_src, ref_buffer + 0, simd);
                createRandsimd_1(dither, &gen_rand, ditherYC2, ditherYC);
                gather1(ycp_buffer + 16, ycp_src, ref_buffer + 8, simd);

                __m128i xYCPRef0, xYCPRef1, xYCPRef2, xYCPRef3;
                __m128i xYCPAvg, xYCPDiff, xYCP, xThreshold, xBase, xMask, xDither;

                xYCPRef0 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +   0));
                xYCPRef1 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +  48));
                xYCPRef2 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +  96));
                xYCPRef3 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer + 144));
                xYCPAvg  = _mm_srai_epi16(_mm_adds_epi16(xTwo,
                    _mm_adds_epi16(_mm_adds_epi16(xYCPRef0, xYCPRef1),
                        _mm_adds_epi16(xYCPRef2, xYCPRef3))), 2);
                xYCP     = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src +  0));
                xYCPDiff = (blur_first) ? abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPAvg))
                    : _mm_max_epi16(_mm_max_epi16(abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef0)),
                        abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef1))),
                        _mm_max_epi16(abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef2)),
                            abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef3))));
                xThreshold = _mm_load_si128((__m128i*)(threshold +  0));
                xMask    = _mm_cmpgt_epi16(xThreshold, xYCPDiff); //(a > b) ? 0xffff : 0x0000;
                xBase    = blendv_epi8_simd(xYCP, xYCPAvg, xMask);  //r = (mask0 & 0xff) ? b : a
                xDither  = _mm_load_si128((__m128i *)(dither +  0));
                xYCP     = _mm_adds_epi16(xBase, xDither);
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst +  0), xYCP);

                gather2(ycp_buffer +  0, ycp_src, ref_buffer + 0, simd);
                createRandsimd_2(dither, ref, &gen_rand, ditherYC2, ditherYC, xRangeYLimit, xRangeXLimit0, xRangeXLimit1, simd);
                gather2(ycp_buffer + 16, ycp_src, ref_buffer + 8, simd);

                xYCPRef0 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +  16));
                xYCPRef1 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +  64));
                xYCPRef2 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer + 112));
                xYCPRef3 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer + 160));
                xYCPAvg  = _mm_srai_epi16(_mm_adds_epi16(xTwo,
                    _mm_adds_epi16(_mm_adds_epi16(xYCPRef0, xYCPRef1),
                        _mm_adds_epi16(xYCPRef2, xYCPRef3))), 2);
                xYCP     = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src + 16));
                xYCPDiff = (blur_first) ? abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPAvg))
                    : _mm_max_epi16(_mm_max_epi16(abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef0)),
                        abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef1))),
                        _mm_max_epi16(abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef2)),
                            abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef3))));
                xThreshold = _mm_load_si128((__m128i*)(threshold +  8));
                xMask    = _mm_cmpgt_epi16(xThreshold, xYCPDiff); //(a > b) ? 0xffff : 0x0000;
                xBase    = blendv_epi8_simd(xYCP, xYCPAvg, xMask);  //r = (mask0 & 0xff) ? b : a
                xDither  = _mm_load_si128((__m128i *)(dither +  8));
                xYCP     = _mm_adds_epi16(xBase, xDither);
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst + 16), xYCP);



                xYCPRef0 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +  32));
                xYCPRef1 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer +  80));
                xYCPRef2 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer + 128));
                xYCPRef3 = _mm_load_si128((__m128i*)((BYTE*)ycp_buffer + 176));
                xYCPAvg  = _mm_srai_epi16(_mm_adds_epi16(xTwo,
                    _mm_adds_epi16(_mm_adds_epi16(xYCPRef0, xYCPRef1),
                        _mm_adds_epi16(xYCPRef2, xYCPRef3))), 2);
                xYCP     = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src + 32));
                xYCPDiff = (blur_first) ? abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPAvg))
                    : _mm_max_epi16(_mm_max_epi16(abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef0)),
                        abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef1))),
                        _mm_max_epi16(abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef2)),
                            abs_epi16_simd(_mm_subs_epi16(xYCP, xYCPRef3))));
                xThreshold = _mm_load_si128((__m128i*)(threshold + 16));
                xMask    = _mm_cmpgt_epi16(xThreshold, xYCPDiff); //(a > b) ? 0xffff : 0x0000;
                xBase    = blendv_epi8_simd(xYCP, xYCPAvg, xMask);  //r = (mask0 & 0xff) ? b : a
                xDither  = _mm_load_si128((__m128i *)(dither + 16));
                xYCP     = _mm_adds_epi16(xBase, xDither);
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst + 32), xYCP);

                i_step = limit_1_to_8(x);
            }
        }
        //最後のライン
        if (y < y_end) {
            PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w + x_start;
            PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w + x_start;
            for (int i_step = 0, x = (x_end - x_start) - 8; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {

                __m128i xYCP0, xYCP1, xYCP2, xDither0, xDither1, xDither2;
                createRandsimd_4(&gen_rand, ditherYC2, ditherYC, xDither0, xDither1, xDither2);

                xYCP0    = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src +  0));
                xYCP1    = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src + 16));
                xYCP2    = _mm_loadu_si128((__m128i*)((BYTE *)ycp_src + 32));
                xYCP0    = _mm_adds_epi16(xYCP0, xDither0);
                xYCP1    = _mm_adds_epi16(xYCP1, xDither1);
                xYCP2    = _mm_adds_epi16(xYCP2, xDither2);
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst +  0), xYCP0);
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst + 16), xYCP1);
                _mm_storeu_si128((__m128i*)((BYTE *)ycp_dst + 32), xYCP2);

                i_step = limit_1_to_8(x);
            }
        }
    }
    _mm_empty();
    band.gen_rand[thread_id] = gen_rand;
}
#endif //#if USE_SSE

#endif //_BANDING_SIMD_H_
