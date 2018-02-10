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

#define USE_SSE2   1
#define USE_SSSE3  0
#define USE_SSE41  0
#define USE_AVX2   0
#define USE_AVX512 0
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <algorithm>
#include <emmintrin.h> //SSE2
#include "banding.h"
#include "xor_rand.h"
#include "filter.h"
#include "banding_simd.h"

void decrease_banding_mode0_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode0_simd<false>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode0_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode0_simd<true>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode1_blur_first_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd<true, false>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode1_blur_first_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd<true, true>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode1_blur_later_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd<false, false>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode1_blur_later_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd<false, true>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode2_blur_first_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd<true, false>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode2_blur_first_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd<true, true>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode2_blur_later_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd<false, false>(thread_id, thread_num, fp, fpip);
}

void decrease_banding_mode2_blur_later_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd<false, true>(thread_id, thread_num, fp, fpip);
}
