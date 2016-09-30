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

#include <Windows.h>
#include <algorithm>
#include <emmintrin.h> //SSE2
#include <smmintrin.h> //SSE4.1
#include "banding.h"
#include "xor_rand.h"
#include "filter.h"
#include "banding_simd.h"

void decrease_banding_mode0_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode0_simd(thread_id, thread_num, fp, fpip, FALSE, SSE2);
}

void decrease_banding_mode0_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode0_simd(thread_id, thread_num, fp, fpip, TRUE, SSE2);
}

void decrease_banding_mode0_p_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode0_simd(thread_id, thread_num, fp, fpip, FALSE, SSSE3|SSE2);
}

void decrease_banding_mode0_i_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode0_simd(thread_id, thread_num, fp, fpip, TRUE, SSSE3|SSE2);
}

void decrease_banding_mode0_p_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode0_simd(thread_id, thread_num, fp, fpip, FALSE, SSE41|SSSE3|SSE2);
}

void decrease_banding_mode0_i_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode0_simd(thread_id, thread_num, fp, fpip, TRUE, SSE41|SSSE3|SSE2);
}

void decrease_banding_mode1_blur_first_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, TRUE, FALSE, SSE2);
}

void decrease_banding_mode1_blur_first_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, TRUE, TRUE, SSE2);
}

void decrease_banding_mode1_blur_later_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, FALSE, FALSE, SSE2);
}

void decrease_banding_mode1_blur_later_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, FALSE, TRUE, SSE2);
}

void decrease_banding_mode1_blur_first_p_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, TRUE, FALSE, SSSE3|SSE2);
}

void decrease_banding_mode1_blur_first_i_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, TRUE, TRUE, SSSE3|SSE2);
}

void decrease_banding_mode1_blur_later_p_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, FALSE, FALSE, SSSE3|SSE2);
}

void decrease_banding_mode1_blur_later_i_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, FALSE, TRUE, SSSE3|SSE2);
}

void decrease_banding_mode1_blur_first_p_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, TRUE, FALSE, SSE41|SSSE3|SSE2);
}

void decrease_banding_mode1_blur_first_i_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, TRUE, TRUE, SSE41|SSSE3|SSE2);
}

void decrease_banding_mode1_blur_later_p_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, FALSE, FALSE, SSE41|SSSE3|SSE2);
}

void decrease_banding_mode1_blur_later_i_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, FALSE, TRUE, SSE41|SSSE3|SSE2);
}

void decrease_banding_mode2_blur_first_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, TRUE, FALSE, SSE2);
}

void decrease_banding_mode2_blur_first_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, TRUE, TRUE, SSE2);
}

void decrease_banding_mode2_blur_later_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, FALSE, FALSE, SSE2);
}

void decrease_banding_mode2_blur_later_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, FALSE, TRUE, SSE2);
}

void decrease_banding_mode2_blur_first_p_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, TRUE, FALSE, SSSE3|SSE2);
}

void decrease_banding_mode2_blur_first_i_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, TRUE, TRUE, SSSE3|SSE2);
}

void decrease_banding_mode2_blur_later_p_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, FALSE, FALSE, SSSE3|SSE2);
}

void decrease_banding_mode2_blur_later_i_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, FALSE, TRUE, SSSE3|SSE2);
}

void decrease_banding_mode2_blur_first_p_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, TRUE, FALSE, SSE41|SSSE3|SSE2);
}

void decrease_banding_mode2_blur_first_i_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, TRUE, TRUE, SSE41|SSSE3|SSE2);
}

void decrease_banding_mode2_blur_later_p_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, FALSE, FALSE, SSE41|SSSE3|SSE2);
}

void decrease_banding_mode2_blur_later_i_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, FALSE, TRUE, SSE41|SSSE3|SSE2);
}
