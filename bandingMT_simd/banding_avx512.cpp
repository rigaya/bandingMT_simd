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

#define USE_AVX512 1
#include <Windows.h>
#include <emmintrin.h> //SSE2
#include <smmintrin.h> //SSE4.1
#include <immintrin.h> //AVX, AVX2, AVX512
#include "banding.h"
#include "xor_rand.h"
#include "filter.h"
#include "banding_simd.h"

#if defined(_MSC_VER) && _MSC_VER >= 1800 && !defined(__AVX__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX or /arch:AVX2 for this file.");
#endif

//実は普通にmemcpyのほうが速いかもだけど気にしない
static void __forceinline avx512_memcpy(BYTE *dst, BYTE *src, int size) {
    if (size < 256) {
        for (int i = 0; i < size; i++)
            dst[i] = src[i];
        return;
    }
    BYTE *dst_fin = dst + size;
    BYTE *dst_aligned_fin = (BYTE *)(((size_t)(dst_fin + 63) & ~63) - 256);
    __m512i y0, y1, y2, y3;
    const int start_align_diff = (int)((size_t)dst & 63);
    if (start_align_diff) {
        y0 = _mm512_loadu_si512((__m512i*)src);
        _mm512_storeu_si512((__m512i*)dst, y0);
        dst += 64 - start_align_diff;
        src += 64 - start_align_diff;
    }
    for (; dst < dst_aligned_fin; dst += 256, src += 256) {
        y0 = _mm512_loadu_si512((__m512i*)(src +   0));
        y1 = _mm512_loadu_si512((__m512i*)(src +  64));
        y2 = _mm512_loadu_si512((__m512i*)(src + 128));
        y3 = _mm512_loadu_si512((__m512i*)(src + 192));
        _mm512_storeu_si512((__m512i*)(dst +   0), y0);
        _mm512_storeu_si512((__m512i*)(dst +  64), y1);
        _mm512_storeu_si512((__m512i*)(dst + 128), y2);
        _mm512_storeu_si512((__m512i*)(dst + 192), y3);
    }
    BYTE *dst_tmp = dst_fin - 256;
    src -= (dst - dst_tmp);
    y0 = _mm512_loadu_si512((__m512i*)(src +  0));
    y1 = _mm512_loadu_si512((__m512i*)(src + 32));
    y2 = _mm512_loadu_si512((__m512i*)(src + 64));
    y3 = _mm512_loadu_si512((__m512i*)(src + 96));
    _mm512_storeu_si512((__m512i*)(dst_tmp +  0), y0);
    _mm512_storeu_si512((__m512i*)(dst_tmp + 32), y1);
    _mm512_storeu_si512((__m512i*)(dst_tmp + 64), y2);
    _mm512_storeu_si512((__m512i*)(dst_tmp + 96), y3);
}

#define zAllBit (_mm512_ternarylogic_epi32(_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), 0xff))
#define zOne512 (_mm512_srli_epi16(zAllBit, 15))
#define zTwo512 (_mm512_slli_epi16(zOne512, 1))
#define zC_16   (_mm512_slli_epi16(zOne512, 4))

static __forceinline __m512i _mm512_multi6_epi32(__m512i a) {
    return _mm512_add_epi32(_mm512_slli_epi32(a, 1), _mm512_slli_epi32(a, 2));
}
static __forceinline __m512i _mm512_multi3_epi32(__m512i a) {
    return _mm512_add_epi32(a, _mm512_add_epi32(a, a));
}

static __forceinline __m512i _mm512_neg_epi32(__m512i y) {
    return _mm512_sub_epi32(_mm512_setzero_si512(), y);
}
static __forceinline __m512i _mm512_neg_epi16(__m512i y) {
    return _mm512_sub_epi16(_mm512_setzero_si512(), y);
}

static __forceinline int limit_1_to_32(int value) {
    int cmp_ret = (value>=32);
    return (cmp_ret<<5) + (value & (0x1f & (~(0-cmp_ret)))) + (value == 0);
}

static __forceinline __m512i apply_field_mask_512(__m512i yRef, BOOL to_lower_byte) {
    __m512i yFeildMask = _mm512_slli_epi16(_mm512_ternarylogic_epi32(_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), 0xff), 1);
    if (!to_lower_byte) //16bitの下位バイトに適用するかどうか
        yFeildMask = _mm512_alignr_epi8(yFeildMask, yFeildMask, 1);
    //負かどうか
    __mmask64 yMaskNeg = _mm512_cmpgt_epi8_mask(_mm512_setzero_si512(), yRef);
    __m512i yFieldMaskHit = _mm512_andnot_si512(yFeildMask, yRef);
    yRef = _mm512_and_si512(yRef, yFeildMask);
    //yFieldMaskHitかつ負なら+2する必要がある
    yRef = _mm512_mask_add_epi8(yRef, yMaskNeg, yRef, _mm512_slli_epi16(yFieldMaskHit, 1));
    return yRef;
}
//mode012共通 ... ref用乱数の見を発生させる
static void __forceinline createRandAVX2_0(BYTE *ref_ptr, xor514x4_t *gen_rand, __m512i zRangeYLimit, __m512i& zRangeXLimit0, __m512i& zRangeXLimit1) {
    __m512i y0, y1;
    __m256i x0, x1;
    __m512i zRange = _mm512_min_epu16(zRangeYLimit, _mm512_min_epu16(zRangeXLimit0, _mm512_max_epi16(zRangeXLimit1, _mm512_setzero_si512())));
    __m512i zRange2 = _mm512_adds_epu16(_mm512_slli_epi16(zRange, 1), zOne512);

    xor514x4(gen_rand);
    x0 = gen_rand->n[6];
    x1 = gen_rand->n[7];
    y0 = _mm512_cvtepu8_epi16(x0);
    y1 = _mm512_cvtepu8_epi16(_mm256_xor_si256(x0, x1));
    y0 = _mm512_mullo_epi16(y0, zRange2);
    y1 = _mm512_mullo_epi16(y1, zRange2);
    y0 = _mm512_srai_epi16(y0, 8);
    y1 = _mm512_srai_epi16(y1, 8);
    y0 = _mm512_or_si512(y0, _mm512_bslli_epi128(y1, 1));
    y0 = _mm512_sub_epi8(y0, _mm512_or_si512(zRange, _mm512_bslli_epi128(zRange, 1)));
    _mm512_store_si512((__m512i*)(ref_ptr), y0);
    
    zRangeXLimit0 = _mm512_add_epi16(zRangeXLimit0, zC_16);
    zRangeXLimit1 = _mm512_sub_epi16(zRangeXLimit1, zC_16);
}
//mode12用 ... dither用乱数のうち2つを発生させる、最後の一つは次で
static void __forceinline createRandAVX2_1(short *dither_ptr, xor514x4_t *gen_rand, const short *ditherYC2, const short *ditherYC) {
    __m512i y0, y1;
    xor514x4(gen_rand);
    y0 = _mm512_cvtepu8_epi16(gen_rand->n[6]);
    y1 = _mm512_cvtepu8_epi16(gen_rand->n[7]);
    y0 = _mm512_mullo_epi16(y0, _mm512_load_si512((__m512i*)(ditherYC2 +  0)));
    y1 = _mm512_mullo_epi16(y1, _mm512_load_si512((__m512i*)(ditherYC2 + 32)));
    y0 = _mm512_srai_epi16(y0, 8);
    y1 = _mm512_srai_epi16(y1, 8);
    y0 = _mm512_sub_epi16(y0, _mm512_load_si512((__m512i*)(ditherYC +  0)));
    y1 = _mm512_sub_epi16(y1, _mm512_load_si512((__m512i*)(ditherYC + 32)));
    _mm512_store_si512((__m512i*)(dither_ptr +  0), y0);
    _mm512_store_si512((__m512i*)(dither_ptr + 32), y1);
}
//mode12用 ... 次で使うref用乱数とdither用乱数の最後の一つを発生させる
static void __forceinline createRandAVX2_2(short *dither_ptr, BYTE *ref_ptr, xor514x4_t *gen_rand, const short *ditherYC2, const short *ditherYC, __m512i zRangeYLimit, __m512i& zRangeXLimit0, __m512i& zRangeXLimit1) {
    __m512i y0, y1;
    __m256i x0, x1;
    __m512i zRange = _mm512_min_epu16(zRangeYLimit, _mm512_min_epu16(zRangeXLimit0, _mm512_max_epi16(zRangeXLimit1, _mm512_setzero_si512())));
    __m512i zRange2 = _mm512_adds_epu16(_mm512_slli_epi16(zRange, 1), zOne512);

    xor514x4(gen_rand);
    x1 = gen_rand->n[7];
    y0 = _mm512_cvtepu8_epi16(x1);
    y0 = _mm512_mullo_epi16(y0, _mm512_load_si512((__m512i*)(ditherYC2 + 64)));
    y0 = _mm512_srai_epi16(y0, 8);
    y0 = _mm512_sub_epi16(y0, _mm512_load_si512((__m512i*)(ditherYC + 64)));
    _mm512_store_si512((__m512i*)(dither_ptr + 64), y0);
    
    x0 = gen_rand->n[6];
    y0 = _mm512_cvtepu8_epi16(x0);
    y1 = _mm512_cvtepu8_epi16(_mm256_xor_si256(x0, x1));
    y0 = _mm512_mullo_epi16(y0, zRange2);
    y1 = _mm512_mullo_epi16(y1, zRange2);
    y0 = _mm512_srai_epi16(y0, 8);
    y1 = _mm512_srai_epi16(y1, 8);
    y0 = _mm512_or_si512(y0, _mm512_bslli_epi128(y1, 1));
    y0 = _mm512_sub_epi8(y0, _mm512_or_si512(zRange, _mm512_bslli_epi128(zRange, 1)));
    _mm512_store_si512((__m512i*)(ref_ptr), y0);
    
    zRangeXLimit0 = _mm512_add_epi16(zRangeXLimit0, zC_16);
    zRangeXLimit1 = _mm512_sub_epi16(zRangeXLimit1, zC_16);
}
//mode0用 ... ref用乱数のみを発生させる
static void __forceinline createRandAVX2_3(BYTE *ref_ptr, xor514x4_t *gen_rand, __m512i zRangeYLimit, __m512i& zRangeXLimit0, __m512i& zRangeXLimit1) {
    __m512i y0, y1;
    __m256i x0, x1;
    __m512i zRange = _mm512_min_epu16(zRangeYLimit, _mm512_min_epu16(zRangeXLimit0, _mm512_max_epi16(zRangeXLimit1, _mm512_setzero_si512())));
    __m512i zRange2 = _mm512_adds_epu16(_mm512_slli_epi16(zRange, 1), zOne512);

    xor514x4(gen_rand);
    x0 = gen_rand->n[6];
    x1 = gen_rand->n[7];
    y0 = _mm512_cvtepu8_epi16(x0);
    y1 = _mm512_cvtepu8_epi16(x1);
    y0 = _mm512_mullo_epi16(y0, zRange2);
    y1 = _mm512_mullo_epi16(y1, zRange2);
    y0 = _mm512_srai_epi16(y0, 8);
    y1 = _mm512_srai_epi16(y1, 8);
    y0 = _mm512_or_si512(y0, _mm512_bslli_epi128(y1, 1));
    y0 = _mm512_sub_epi8(y0, _mm512_or_si512(zRange, _mm512_bslli_epi128(zRange, 1)));
    _mm512_store_si512((__m512i*)(ref_ptr), y0);
    
    zRangeXLimit0 = _mm512_add_epi16(zRangeXLimit0, zC_16);
    zRangeXLimit1 = _mm512_sub_epi16(zRangeXLimit1, zC_16);
}
//mode12用 ... dither用乱数のみを発生させy0,y1,y2で返す
static void __forceinline createRandAVX2_4(xor514x4_t *gen_rand, const short *ditherYC2, const short *ditherYC, __m512i& y0, __m512i& y1, __m512i& y2) {
    xor514x4(gen_rand);
    y0 = _mm512_cvtepu8_epi16(gen_rand->n[6]);
    y1 = _mm512_cvtepu8_epi16(gen_rand->n[7]);
    y2 = _mm512_cvtepu8_epi16(_mm256_xor_si256(gen_rand->n[6], gen_rand->n[7]));
    y0 = _mm512_mullo_epi16(y0, _mm512_load_si512((__m512i*)(ditherYC2 +  0)));
    y1 = _mm512_mullo_epi16(y1, _mm512_load_si512((__m512i*)(ditherYC2 + 32)));
    y2 = _mm512_mullo_epi16(y2, _mm512_load_si512((__m512i*)(ditherYC2 + 64)));
    y0 = _mm512_srai_epi16(y0, 8);
    y1 = _mm512_srai_epi16(y1, 8);
    y2 = _mm512_srai_epi16(y2, 8);
    y0 = _mm512_sub_epi16(y0, _mm512_load_si512((__m512i*)(ditherYC +  0)));
    y1 = _mm512_sub_epi16(y1, _mm512_load_si512((__m512i*)(ditherYC + 32)));
    y2 = _mm512_sub_epi16(y2, _mm512_load_si512((__m512i*)(ditherYC + 64)));
}

alignas(64) static const uint16_t PACK_YC48_SHUFFLE_AVX512[48] = {
    0,  1,  2,  4,  5,  6,  8,  9,  10, 12, 13, 14, 16, 17, 18, 20,
    21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41,
    42, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62
};

static void __forceinline pack_yc48_1(__m512i& yGather0, const __m512i& yGather1) {
    __m512i zGatherIdx = _mm512_load_si512((__m512i*)PACK_YC48_SHUFFLE_AVX512);
    yGather0 = _mm512_permutex2var_epi16(yGather0, zGatherIdx, yGather1);
}
static void __forceinline pack_yc48_2(__m512i& yGather1, __m512i& yGather2, __m512i yGather3) {
    __m512i zGatherIdx0 = _mm512_loadu_si512((__m512i*)(PACK_YC48_SHUFFLE_AVX512 +  8));
    __m512i zGatherIdx1 = _mm512_loadu_si512((__m512i*)(PACK_YC48_SHUFFLE_AVX512 + 16));
    yGather1 = _mm512_permutex2var_epi16(yGather1, zGatherIdx0, yGather2);
    yGather2 = _mm512_permutex2var_epi16(yGather2, zGatherIdx1, yGather3);
}

template<bool process_per_field>
static void __forceinline decrease_banding_mode0_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    const int sample_mode = 0;
    const int max_w  = fpip->max_w;
    const int width = fpip->w;
    const int height = fpip->h;
    const int seed    = fp->track[7];
    const int ditherY = fp->track[4];
    const int ditherC = fp->track[5];
    const int  rand_each_frame = fp->check[1];
    const int  blur_first      = fp->check[0];
    const BYTE range           = fp->track[0];
    const int  threshold_y     = fp->track[1] << (!(sample_mode && blur_first) + 1);
    const int  threshold_cb    = fp->track[2] << (!(sample_mode && blur_first) + 1);
    const int  threshold_cr    = fp->track[3] << (!(sample_mode && blur_first) + 1);
    const int  b_start = (band.block_count_x * band.block_count_y *  thread_id) / thread_num;
    const int  b_end   = (band.block_count_x * band.block_count_y * (thread_id+1)) / thread_num;
    __m512i yRefMulti  = _mm512_unpacklo_epi16(_mm512_set1_epi16(max_w), zOne512);
    __m512i yGather0, yGather1, yGather2, yGather3;
    
    alignas(64) BYTE   ref[64];
    alignas(64) short  threshold[96];

    for (int i = 0; i < 32; i++) {
        threshold[3*i+0] = threshold_y;
        threshold[3*i+1] = threshold_cb;
        threshold[3*i+2] = threshold_cr;
    }
    xor514x4_t gen_rand;
    if (!rand_each_frame) {
        xor514x4_init(&gen_rand, seed + (b_start << 4));
    } else {
        gen_rand = band.gen_rand_avx512[thread_id];
    }

    for (int ib = b_start; ib < b_end; ib++) {
        int x_start, x_end, y_start, y_end;
        band_get_block_range(ib, width, height, &x_start, &x_end, &y_start, &y_end);
        int y;
        const int y_main_end = min(y_end, height - 1);
        for (y = y_start; y < y_main_end; y++) {
            PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w + x_start;
            PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w + x_start;
            __m512i zRangeYLimit = _mm512_set1_epi16(min3(range, y, height - y - 1));
            __m512i zRangeXLimit0 = _mm512_add_epi16(_mm512_load_si512((__m512i*)x_range_offset), _mm512_set1_epi16(x_start));
            __m512i zRangeXLimit1 = _mm512_sub_epi16(_mm512_set1_epi16(width - x_start - 1), _mm512_load_si512((__m512i*)x_range_offset));
            createRandAVX2_0(ref, &gen_rand, zRangeYLimit, zRangeXLimit0, zRangeXLimit1);
            for (int i_step = 0, x = (x_end - x_start) - 32; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {
                __m512i yRef = _mm512_loadu_si512((__m512i*)ref);
                if (process_per_field) {
                    __m512i yFeildMask = _mm512_slli_epi16(zAllBit, 1);
                    yRef = _mm512_and_si512(yRef, yFeildMask);
                }
                __m512i yRefUpper = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(yRef, 1));
                __m512i yRefLower = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(yRef));

                yRefLower = _mm512_multi3_epi32(_mm512_add_epi32(_mm512_madd_epi16(yRefLower, yRefMulti), _mm512_load_si512((__m512i*)&ref_offset[ 0])));
                yRefUpper = _mm512_multi3_epi32(_mm512_add_epi32(_mm512_madd_epi16(yRefUpper, yRefMulti), _mm512_load_si512((__m512i*)&ref_offset[16])));

                yGather1 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefLower, 1), (void *)ycp_src, 2);
                yGather0 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefLower),       (void *)ycp_src, 2);
                pack_yc48_1(yGather0, yGather1);

                __m512i yYCPRef0, yYCPDiff, yYCP, yThreshold, yBase;

                yYCPRef0 = yGather0;
                yYCP     = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src +  0));
                yYCPDiff = _mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef0));
                yThreshold = _mm512_load_si512((__m512i*)(threshold +  0));
                yBase      = _mm512_mask_mov_epi16(yYCP, _mm512_cmpgt_epi16_mask(yThreshold, yYCPDiff), yYCPRef0);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst +  0), yBase);

                yGather3 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefUpper, 1), (void *)ycp_src, 2);
                yGather2 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefUpper),       (void *)ycp_src, 2);
                pack_yc48_2(yGather1, yGather2, yGather3);
                createRandAVX2_3(ref, &gen_rand, zRangeYLimit, zRangeXLimit0, zRangeXLimit1);

                yYCPRef0 = yGather1;
                yYCP     = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src + 64));
                yYCPDiff = _mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef0));
                yThreshold = _mm512_load_si512((__m512i*)(threshold +  32));
                yBase      = _mm512_mask_mov_epi16(yYCP, _mm512_cmpgt_epi16_mask(yThreshold, yYCPDiff), yYCPRef0);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst + 64), yBase);

                yYCPRef0 = yGather2;
                yYCP     = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src + 128));
                yYCPDiff = _mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef0));
                yThreshold = _mm512_load_si512((__m512i*)(threshold + 64));
                yBase      = _mm512_mask_mov_epi16(yYCP, _mm512_cmpgt_epi16_mask(yThreshold, yYCPDiff), yYCPRef0);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst + 128), yBase);

                i_step = limit_1_to_32(x);
            }
        }
        //最後のライン
        if (y < y_end)
            avx512_memcpy((BYTE *)(fpip->ycp_temp + y * max_w + x_start), (BYTE *)(fpip->ycp_edit + y * max_w + x_start), (x_end - x_start) * 6);
    }
    band.gen_rand_avx512[thread_id] = gen_rand;
}

void decrease_banding_mode0_p_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode0_avx512<false>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode0_i_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode0_avx512<true>(thread_id, thread_num, fp, fpip);
}

template<bool blur_first, bool process_per_field>
static void __forceinline decrease_banding_mode1_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    const int sample_mode = 1;
    const int max_w  = fpip->max_w;
    const int width = fpip->w;
    const int height = fpip->h;
    const int seed    = fp->track[7];
    const int ditherY = fp->track[4];
    const int ditherC = fp->track[5];
    const int  rand_each_frame = fp->check[1];
    const BYTE range           = fp->track[0];
    const int  threshold_y     = fp->track[1] << (!(sample_mode && blur_first) + 1);
    const int  threshold_cb    = fp->track[2] << (!(sample_mode && blur_first) + 1);
    const int  threshold_cr    = fp->track[3] << (!(sample_mode && blur_first) + 1);
    const int  b_start = (band.block_count_x * band.block_count_y *  thread_id) / thread_num;
    const int  b_end   = (band.block_count_x * band.block_count_y * (thread_id+1)) / thread_num;
    __m512i yRefMulti  = _mm512_unpacklo_epi16(_mm512_set1_epi16(max_w), zOne512);
    __m512i yGather0, yGather1, yGather2, yGather3, yGather4, yGather5, yGather6, yGather7;
    
    alignas(64) BYTE   ref[64];
    alignas(64) short  dither[96];
    alignas(64) short  ditherYC[96];
    alignas(64) short  ditherYC2[96];
    alignas(64) short  threshold[96];

    for (int i = 0; i < 32; i++) {
        threshold[3*i+0] = threshold_y;
        threshold[3*i+1] = threshold_cb;
        threshold[3*i+2] = threshold_cr;
    }
    for (int i = 0; i < 32; i++) {
        ditherYC[3*i+0] = ditherY;
        ditherYC[3*i+1] = ditherC;
        ditherYC[3*i+2] = ditherC;
    }
    {
        __m512i y0 = _mm512_load_si512((__m512i *)(ditherYC +  0));
        __m512i y1 = _mm512_load_si512((__m512i *)(ditherYC + 32));
        __m512i y2 = _mm512_load_si512((__m512i *)(ditherYC + 64));
        y0 = _mm512_slli_epi16(y0, 1);
        y1 = _mm512_slli_epi16(y1, 1);
        y2 = _mm512_slli_epi16(y2, 1);
        y0 = _mm512_add_epi16(y0, zOne512);
        y1 = _mm512_add_epi16(y1, zOne512);
        y2 = _mm512_add_epi16(y2, zOne512);
        _mm512_store_si512((__m512i *)(ditherYC2 +  0), y0);
        _mm512_store_si512((__m512i *)(ditherYC2 + 32), y1);
        _mm512_store_si512((__m512i *)(ditherYC2 + 64), y2);
    }
    xor514x4_t gen_rand;
    if (!rand_each_frame) {
        xor514x4_init(&gen_rand, seed + (b_start << 4));
    } else {
        gen_rand = band.gen_rand_avx512[thread_id];
    }

    for (int ib = b_start; ib < b_end; ib++) {
        int x_start, x_end, y_start, y_end;
        band_get_block_range(ib, width, height, &x_start, &x_end, &y_start, &y_end);
        int y;
        const int y_main_end = min(y_end, height - 1);
        for (y = y_start; y < y_main_end; y++) {
            PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w + x_start;
            PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w + x_start;
            __m512i zRangeYLimit = _mm512_set1_epi16(min3(range, y, height - y - 1));
            __m512i zRangeXLimit0 = _mm512_add_epi16(_mm512_load_si512((__m512i*)x_range_offset), _mm512_set1_epi16(x_start));
            __m512i zRangeXLimit1 = _mm512_sub_epi16(_mm512_set1_epi16(width - x_start - 1), _mm512_load_si512((__m512i*)x_range_offset));
            createRandAVX2_0(ref, &gen_rand, zRangeYLimit, zRangeXLimit0, zRangeXLimit1);
            for (int i_step = 0, x = (x_end - x_start) - 32; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {
                __m512i yRef = _mm512_loadu_si512((__m512i*)ref);
                if (process_per_field) {
                    yRef = apply_field_mask_512(yRef, TRUE);
                }
                __m512i yRefUpper = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(yRef, 1));
                __m512i yRefLower = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(yRef));

                yRefLower = _mm512_madd_epi16(yRefLower, yRefMulti);
                yRefUpper = _mm512_madd_epi16(yRefUpper, yRefMulti);

                __m512i yRefLowerP = _mm512_multi3_epi32(_mm512_add_epi32(yRefLower, _mm512_load_si512((__m512i*)&ref_offset[0])));
                __m512i yRefLowerM = _mm512_multi3_epi32(_mm512_sub_epi32(_mm512_load_si512((__m512i*)&ref_offset[0]), yRefLower));

                yGather1 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefLowerP, 1), (void *)ycp_src, 2);
                yGather0 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefLowerP),       (void *)ycp_src, 2);
                yGather5 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefLowerM, 1), (void *)ycp_src, 2);
                yGather4 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefLowerM),       (void *)ycp_src, 2);

                pack_yc48_1(yGather0, yGather1);
                pack_yc48_1(yGather4, yGather5);
                createRandAVX2_1(dither, &gen_rand, ditherYC2, ditherYC);

                __m512i yYCPRef0, yYCPRef1, xYCPAvg, yYCPDiff, yYCP, yThreshold, yBase, yDither;
                __mmask32 mBlend;

                yYCPRef0 = yGather0;
                yYCPRef1 = yGather4;
                xYCPAvg  = _mm512_srai_epi16(_mm512_adds_epi16(_mm512_adds_epi16(yYCPRef0, yYCPRef1), zOne512), 1);
                yYCP     = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src +  0));
                yYCPDiff = (blur_first) ? _mm512_abs_epi16(_mm512_subs_epi16(yYCP, xYCPAvg))
                                        : _mm512_max_epi16(_mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef0)),
                                                           _mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef1)));
                yThreshold = _mm512_load_si512((__m512i*)(threshold +  0));
                mBlend     = _mm512_cmpgt_epi16_mask(yThreshold, yYCPDiff);
                yBase      = _mm512_mask_mov_epi16(yYCP, mBlend, xYCPAvg);
                yDither  = _mm512_load_si512((__m512i *)(dither +  0));
                yYCP     = _mm512_adds_epi16(yBase, yDither);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst +  0), yYCP);

                __m512i yRefUpperP = _mm512_multi3_epi32(_mm512_add_epi32(yRefUpper, _mm512_load_si512((__m512i*)&ref_offset[16])));
                __m512i yRefUpperM = _mm512_multi3_epi32(_mm512_sub_epi32(_mm512_load_si512((__m512i*)&ref_offset[16]), yRefUpper));

                yGather3 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefUpperP, 1), (void *)ycp_src, 2);
                yGather2 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefUpperP),       (void *)ycp_src, 2);
                yGather7 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefUpperM, 1), (void *)ycp_src, 2);
                yGather6 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefUpperM),       (void *)ycp_src, 2);

                pack_yc48_2(yGather1, yGather2, yGather3);
                pack_yc48_2(yGather5, yGather6, yGather7);

                createRandAVX2_2(dither, ref, &gen_rand, ditherYC2, ditherYC, zRangeYLimit, zRangeXLimit0, zRangeXLimit1);

                yYCPRef0 = yGather1;
                yYCPRef1 = yGather5;
                xYCPAvg  = _mm512_srai_epi16(_mm512_adds_epi16(_mm512_adds_epi16(yYCPRef0, yYCPRef1), zOne512), 1);
                yYCP     = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src + 64));
                yYCPDiff = (blur_first) ? _mm512_abs_epi16(_mm512_subs_epi16(yYCP, xYCPAvg))
                                        : _mm512_max_epi16(_mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef0)),
                                                           _mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef1)));
                yThreshold = _mm512_load_si512((__m512i*)(threshold + 32));
                mBlend     = _mm512_cmpgt_epi16_mask(yThreshold, yYCPDiff);
                yBase      = _mm512_mask_mov_epi16(yYCP, mBlend, xYCPAvg);
                yDither  = _mm512_load_si512((__m512i *)(dither + 32));
                yYCP     = _mm512_adds_epi16(yBase, yDither);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst + 64), yYCP);

                yYCPRef0 = yGather2;
                yYCPRef1 = yGather6;
                xYCPAvg  = _mm512_srai_epi16(_mm512_adds_epi16(_mm512_adds_epi16(yYCPRef0, yYCPRef1), zOne512), 1);
                yYCP     = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src + 128));
                yYCPDiff = (blur_first) ? _mm512_abs_epi16(_mm512_subs_epi16(yYCP, xYCPAvg))
                                        : _mm512_max_epi16(_mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef0)),
                                                           _mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef1)));
                yThreshold = _mm512_load_si512((__m512i*)(threshold + 64));
                mBlend     = _mm512_cmpgt_epi16_mask(yThreshold, yYCPDiff);
                yBase      = _mm512_mask_mov_epi16(yYCP, mBlend, xYCPAvg);
                yDither  = _mm512_load_si512((__m512i *)(dither + 64));
                yYCP     = _mm512_adds_epi16(yBase, yDither);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst + 128), yYCP);

                i_step = limit_1_to_32(x);
            }
        }
        //最後のライン
        if (y < y_end) {
            PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w + x_start;
            PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w + x_start;
            for (int i_step = 0, x = (x_end - x_start) - 32; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {

                __m512i yYCP0, yYCP1, yYCP2, yDither0, yDither1, yDither2;
                createRandAVX2_4(&gen_rand, ditherYC2, ditherYC, yDither0, yDither1, yDither2);

                yYCP0    = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src +   0));
                yYCP1    = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src +  64));
                yYCP2    = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src + 128));
                yYCP0    = _mm512_adds_epi16(yYCP0, yDither0);
                yYCP1    = _mm512_adds_epi16(yYCP1, yDither1);
                yYCP2    = _mm512_adds_epi16(yYCP2, yDither2);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst +   0), yYCP0);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst +  64), yYCP1);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst + 128), yYCP2);

                i_step = limit_1_to_32(x);
            }
        }
    }
    band.gen_rand_avx512[thread_id] = gen_rand;
}

void decrease_banding_mode1_blur_first_p_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_avx512<true, false>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode1_blur_first_i_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_avx512<true, true>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode1_blur_later_p_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_avx512<false, false>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode1_blur_later_i_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_avx512<false, true>(thread_id, thread_num, fp, fpip);
}

template<bool blur_first, bool process_per_field>
static void __forceinline decrease_banding_mode2_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    const int sample_mode = 2;
    const int max_w  = fpip->max_w;
    const int width = fpip->w;
    const int height = fpip->h;
    const int seed    = fp->track[7];
    const int ditherY = fp->track[4];
    const int ditherC = fp->track[5];
    const int  rand_each_frame = fp->check[1];
    const BYTE range           = fp->track[0];
    const int  threshold_y     = fp->track[1] << (!(sample_mode && blur_first) + 1);
    const int  threshold_cb    = fp->track[2] << (!(sample_mode && blur_first) + 1);
    const int  threshold_cr    = fp->track[3] << (!(sample_mode && blur_first) + 1);
    const int  b_start = (band.block_count_x * band.block_count_y *  thread_id) / thread_num;
    const int  b_end   = (band.block_count_x * band.block_count_y * (thread_id+1)) / thread_num;
    __m512i yRefMulti  = _mm512_unpacklo_epi16(_mm512_set1_epi16(max_w), zOne512);
    __m512i yRefMulti2 = _mm512_unpacklo_epi16(zOne512, _mm512_set1_epi16(-max_w));
    
    alignas(64) BYTE      ref[64];
    alignas(64) short     dither[96];
    alignas(64) short     ditherYC[96];
    alignas(64) short     ditherYC2[96];
    alignas(64) short     threshold[96];

    for (int i = 0; i < 32; i++) {
        threshold[3*i+0] = threshold_y;
        threshold[3*i+1] = threshold_cb;
        threshold[3*i+2] = threshold_cr;
    }
    for (int i = 0; i < 32; i++) {
        ditherYC[3*i+0] = ditherY;
        ditherYC[3*i+1] = ditherC;
        ditherYC[3*i+2] = ditherC;
    }
    {
        __m512i y0 = _mm512_load_si512((__m512i *)(ditherYC +  0));
        __m512i y1 = _mm512_load_si512((__m512i *)(ditherYC + 32));
        __m512i y2 = _mm512_load_si512((__m512i *)(ditherYC + 64));
        y0 = _mm512_slli_epi16(y0, 1);
        y1 = _mm512_slli_epi16(y1, 1);
        y2 = _mm512_slli_epi16(y2, 1);
        y0 = _mm512_add_epi16(y0, zOne512);
        y1 = _mm512_add_epi16(y1, zOne512);
        y2 = _mm512_add_epi16(y2, zOne512);
        _mm512_store_si512((__m512i *)(ditherYC2 +  0), y0);
        _mm512_store_si512((__m512i *)(ditherYC2 + 32), y1);
        _mm512_store_si512((__m512i *)(ditherYC2 + 64), y2);
    }
    xor514x4_t gen_rand;
    if (!rand_each_frame) {
        xor514x4_init(&gen_rand, seed + (b_start << 4));
    } else {
        gen_rand = band.gen_rand_avx512[thread_id];
    }


    for (int ib = b_start; ib < b_end; ib++) {
        int x_start, x_end, y_start, y_end;
        band_get_block_range(ib, width, height, &x_start, &x_end, &y_start, &y_end);
        int y;
        const int y_main_end = min(y_end, height - 1);
        for (y = y_start; y < y_main_end; y++) {
            PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w + x_start;
            PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w + x_start;
            __m512i zRangeYLimit = _mm512_set1_epi16(min3(range, y, height - y - 1));
            __m512i zRangeXLimit0 = _mm512_add_epi16(_mm512_load_si512((__m512i*)x_range_offset), _mm512_set1_epi16(x_start));
            __m512i zRangeXLimit1 = _mm512_sub_epi16(_mm512_set1_epi16(width - x_start - 1), _mm512_load_si512((__m512i*)x_range_offset));
            createRandAVX2_0(ref, &gen_rand, zRangeYLimit, zRangeXLimit0, zRangeXLimit1);
            for (int i_step = 0, x = (x_end - x_start) - 32; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {
                __m512i yRef = _mm512_loadu_si512((__m512i*)ref);

                __m512i yRefLowerP1, yRefLowerM1, yRefLowerP2, yRefLowerM2;
                __m512i yRefUpperP1, yRefUpperM1, yRefUpperP2, yRefUpperM2;

                if (process_per_field) {
                    __m512i yRef2 = apply_field_mask_512(yRef, TRUE);
                    __m512i yRefUpper = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(yRef2, 1));
                    __m512i yRefLower = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(yRef2));

                    yRefLowerP1 = _mm512_multi3_epi32(_mm512_add_epi32(_mm512_madd_epi16(yRefLower, yRefMulti), _mm512_load_si512((__m512i*)&ref_offset[ 0])));
                    yRefUpperP1 = _mm512_multi3_epi32(_mm512_add_epi32(_mm512_madd_epi16(yRefUpper, yRefMulti), _mm512_load_si512((__m512i*)&ref_offset[16])));
                    yRefLowerM1 = _mm512_multi3_epi32(_mm512_sub_epi32(_mm512_load_si512((__m512i*)&ref_offset[ 0]), _mm512_madd_epi16(yRefLower, yRefMulti)));
                    yRefUpperM1 = _mm512_multi3_epi32(_mm512_sub_epi32(_mm512_load_si512((__m512i*)&ref_offset[16]), _mm512_madd_epi16(yRefUpper, yRefMulti)));

                    yRef2 = apply_field_mask_512(yRef, FALSE);
                    yRefUpper = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(yRef2, 1));
                    yRefLower = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(yRef2));

                    yRefLowerP2 = _mm512_multi3_epi32(_mm512_add_epi32(_mm512_madd_epi16(yRefLower, yRefMulti2), _mm512_load_si512((__m512i*)&ref_offset[ 0])));
                    yRefUpperP2 = _mm512_multi3_epi32(_mm512_add_epi32(_mm512_madd_epi16(yRefUpper, yRefMulti2), _mm512_load_si512((__m512i*)&ref_offset[16])));
                    yRefLowerM2 = _mm512_multi3_epi32(_mm512_sub_epi32(_mm512_load_si512((__m512i*)&ref_offset[ 0]), _mm512_madd_epi16(yRefLower, yRefMulti2)));
                    yRefUpperM2 = _mm512_multi3_epi32(_mm512_sub_epi32(_mm512_load_si512((__m512i*)&ref_offset[16]), _mm512_madd_epi16(yRefUpper, yRefMulti2)));
                } else {
                    __m512i yRefUpper = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(yRef, 1));
                    __m512i yRefLower = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(yRef));

                    yRefLowerP1 = _mm512_multi3_epi32(_mm512_add_epi32(_mm512_madd_epi16(yRefLower, yRefMulti), _mm512_load_si512((__m512i*)&ref_offset[0])));
                    yRefUpperP1 = _mm512_multi3_epi32(_mm512_add_epi32(_mm512_madd_epi16(yRefUpper, yRefMulti), _mm512_load_si512((__m512i*)&ref_offset[16])));
                    yRefLowerM1 = _mm512_multi3_epi32(_mm512_sub_epi32(_mm512_load_si512((__m512i*)&ref_offset[ 0]), _mm512_madd_epi16(yRefLower, yRefMulti)));
                    yRefUpperM1 = _mm512_multi3_epi32(_mm512_sub_epi32(_mm512_load_si512((__m512i*)&ref_offset[16]), _mm512_madd_epi16(yRefUpper, yRefMulti)));

                    yRefLowerP2 = _mm512_multi3_epi32(_mm512_add_epi32(_mm512_madd_epi16(yRefLower, yRefMulti2), _mm512_load_si512((__m512i*)&ref_offset[0])));
                    yRefUpperP2 = _mm512_multi3_epi32(_mm512_add_epi32(_mm512_madd_epi16(yRefUpper, yRefMulti2), _mm512_load_si512((__m512i*)&ref_offset[16])));
                    yRefLowerM2 = _mm512_multi3_epi32(_mm512_sub_epi32(_mm512_load_si512((__m512i*)&ref_offset[ 0]), _mm512_madd_epi16(yRefLower, yRefMulti2)));
                    yRefUpperM2 = _mm512_multi3_epi32(_mm512_sub_epi32(_mm512_load_si512((__m512i*)&ref_offset[16]), _mm512_madd_epi16(yRefUpper, yRefMulti2)));
                }

                __m512i yGatherLP10, yGatherLP11, yGatherLM10, yGatherLM11, yGatherLP20, yGatherLP21, yGatherLM20, yGatherLM21;
                __m512i yGatherUP10, yGatherUP11, yGatherUM10, yGatherUM11, yGatherUP20, yGatherUP21, yGatherUM20, yGatherUM21;

                yGatherLP10 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefLowerP1, 1), (void *)ycp_src, 2);
                yGatherLP11 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefLowerP1),       (void *)ycp_src, 2);
                yGatherLM10 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefLowerM1, 1), (void *)ycp_src, 2);
                yGatherLM11 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefUpperM1),       (void *)ycp_src, 2);

                yGatherLP20 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefLowerP2, 1), (void *)ycp_src, 2);
                yGatherLP21 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefLowerP2),       (void *)ycp_src, 2);
                yGatherLM20 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefLowerM2, 1), (void *)ycp_src, 2);
                yGatherLM21 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefUpperM2),       (void *)ycp_src, 2);

                pack_yc48_1(yGatherLP10, yGatherLP11);
                pack_yc48_1(yGatherLM10, yGatherLM11);
                pack_yc48_1(yGatherLP20, yGatherLP21);
                pack_yc48_1(yGatherLM20, yGatherLM21);

                createRandAVX2_1(dither, &gen_rand, ditherYC2, ditherYC);

                __m512i yYCPRef0, yYCPRef1, xYCPRef2, xYCPRef3;
                __m512i xYCPAvg, yYCPDiff, yYCP, yThreshold, yBase, yDither;
                __mmask32 mBlend;

                yYCPRef0 = yGatherLP10;
                yYCPRef1 = yGatherLM10;
                xYCPRef2 = yGatherLP20;
                xYCPRef3 = yGatherLM20;
                xYCPAvg  = _mm512_srai_epi16(_mm512_adds_epi16(zTwo512,
                                             _mm512_adds_epi16(_mm512_adds_epi16(yYCPRef0, yYCPRef1),
                                                               _mm512_adds_epi16(xYCPRef2, xYCPRef3))), 2);
                yYCP     = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src +  0));
                yYCPDiff = (blur_first) ? _mm512_abs_epi16(_mm512_subs_epi16(yYCP, xYCPAvg))
                                        : _mm512_max_epi16(_mm512_max_epi16(_mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef0)),
                                                                            _mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef1))),
                                                           _mm512_max_epi16(_mm512_abs_epi16(_mm512_subs_epi16(yYCP, xYCPRef2)),
                                                                            _mm512_abs_epi16(_mm512_subs_epi16(yYCP, xYCPRef3))));
                yThreshold = _mm512_load_si512((__m512i*)(threshold +  0));
                mBlend     = _mm512_cmpgt_epi16_mask(yThreshold, yYCPDiff);
                yBase      = _mm512_mask_mov_epi16(yYCP, mBlend, xYCPAvg);
                yDither  = _mm512_load_si512((__m512i *)(dither +  0));
                yYCP     = _mm512_adds_epi16(yBase, yDither);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst +  0), yYCP);

                createRandAVX2_2(dither, ref, &gen_rand, ditherYC2, ditherYC, zRangeYLimit, zRangeXLimit0, zRangeXLimit1);
                
                yGatherUP10 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefUpperP1, 1), (void *)ycp_src, 2);
                yGatherUP11 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefUpperP1),       (void *)ycp_src, 2);
                yGatherUM10 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefUpperM1, 1), (void *)ycp_src, 2);
                yGatherUM11 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefUpperM1),       (void *)ycp_src, 2);

                yGatherUP20 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefUpperP2, 1), (void *)ycp_src, 2);
                yGatherUP21 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefUpperP2),       (void *)ycp_src, 2);
                yGatherUM20 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(yRefUpperM2, 1), (void *)ycp_src, 2);
                yGatherUM21 = _mm512_i32gather_epi64(_mm512_castsi512_si256(yRefUpperM2),       (void *)ycp_src, 2);

                pack_yc48_2(yGatherLP11, yGatherUP10, yGatherUP11);
                pack_yc48_2(yGatherLM11, yGatherUM10, yGatherUM11);
                pack_yc48_2(yGatherLP21, yGatherUP20, yGatherUP21);
                pack_yc48_2(yGatherLM21, yGatherUM20, yGatherUM21);

                yYCPRef0 = yGatherLP11;
                yYCPRef1 = yGatherLM11;
                xYCPRef2 = yGatherLP21;
                xYCPRef3 = yGatherLM21;
                xYCPAvg  = _mm512_srai_epi16(_mm512_adds_epi16(zTwo512,
                                             _mm512_adds_epi16(_mm512_adds_epi16(yYCPRef0, yYCPRef1),
                                                               _mm512_adds_epi16(xYCPRef2, xYCPRef3))), 2);
                yYCP     = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src + 64));
                yYCPDiff = (blur_first) ? _mm512_abs_epi16(_mm512_subs_epi16(yYCP, xYCPAvg))
                                        : _mm512_max_epi16(_mm512_max_epi16(_mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef0)),
                                                                            _mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef1))),
                                                           _mm512_max_epi16(_mm512_abs_epi16(_mm512_subs_epi16(yYCP, xYCPRef2)),
                                                                            _mm512_abs_epi16(_mm512_subs_epi16(yYCP, xYCPRef3))));
                yThreshold = _mm512_load_si512((__m512i*)(threshold + 32));
                mBlend     = _mm512_cmpgt_epi16_mask(yThreshold, yYCPDiff);
                yBase      = _mm512_mask_mov_epi16(yYCP, mBlend, xYCPAvg);
                yDither  = _mm512_load_si512((__m512i *)(dither + 32));
                yYCP     = _mm512_adds_epi16(yBase, yDither);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst + 64), yYCP);



                yYCPRef0 = yGatherUP10;
                yYCPRef1 = yGatherUM10;
                xYCPRef2 = yGatherUP20;
                xYCPRef3 = yGatherUM20;
                xYCPAvg  = _mm512_srai_epi16(_mm512_adds_epi16(zTwo512,
                                             _mm512_adds_epi16(_mm512_adds_epi16(yYCPRef0, yYCPRef1),
                                                               _mm512_adds_epi16(xYCPRef2, xYCPRef3))), 2);
                yYCP     = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src + 128));
                yYCPDiff = (blur_first) ? _mm512_abs_epi16(_mm512_subs_epi16(yYCP, xYCPAvg))
                                        : _mm512_max_epi16(_mm512_max_epi16(_mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef0)),
                                                                            _mm512_abs_epi16(_mm512_subs_epi16(yYCP, yYCPRef1))),
                                                           _mm512_max_epi16(_mm512_abs_epi16(_mm512_subs_epi16(yYCP, xYCPRef2)),
                                                                            _mm512_abs_epi16(_mm512_subs_epi16(yYCP, xYCPRef3))));
                yThreshold = _mm512_load_si512((__m512i*)(threshold + 64));
                mBlend     = _mm512_cmpgt_epi16_mask(yThreshold, yYCPDiff);
                yBase      = _mm512_mask_mov_epi16(yYCP, mBlend, xYCPAvg);
                yDither  = _mm512_load_si512((__m512i *)(dither + 64));
                yYCP     = _mm512_adds_epi16(yBase, yDither);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst + 128), yYCP);

                i_step = limit_1_to_32(x);
            }
        }
        //最後のライン
        if (y < y_end) {
            PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w + x_start;
            PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w + x_start;
            for (int i_step = 0, x = (x_end - x_start) - 32; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {

                __m512i yYCP0, yYCP1, yYCP2, yDither0, yDither1, yDither2;
                createRandAVX2_4(&gen_rand, ditherYC2, ditherYC, yDither0, yDither1, yDither2);

                yYCP0    = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src +   0));
                yYCP1    = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src +  64));
                yYCP2    = _mm512_loadu_si512((__m512i*)((BYTE *)ycp_src + 128));
                yYCP0    = _mm512_adds_epi16(yYCP0, yDither0);
                yYCP1    = _mm512_adds_epi16(yYCP1, yDither1);
                yYCP2    = _mm512_adds_epi16(yYCP2, yDither2);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst +   0), yYCP0);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst +  64), yYCP1);
                _mm512_storeu_si512((__m512i*)((BYTE *)ycp_dst + 128), yYCP2);

                i_step = limit_1_to_32(x);
            }
        }
    }
    _mm256_zeroupper();
    _mm_empty();
    band.gen_rand_avx512[thread_id] = gen_rand;
}

void decrease_banding_mode2_blur_first_p_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_avx512<true, false>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode2_blur_first_i_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_avx512<true, false>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode2_blur_later_p_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_avx512<true, false>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode2_blur_later_i_avx512(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_avx512<true, false>(thread_id, thread_num, fp, fpip);
}
