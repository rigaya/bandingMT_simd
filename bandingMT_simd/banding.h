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
#ifndef _BANDING_H_
#define _BANDING_H_

#include "filter.h"
#include "xor_rand.h"

#ifdef DEFINE_GLOBAL
    #define EXTERN
#else
    #define EXTERN extern
#endif //DEFINE_GLOBAL

#define BANDING_PERF_CHECK 0

typedef void (*func_generate_dither_buffer)(int thread_id, int thread_num, int seed);
typedef void (*func_decrease_banding)(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);

void generate_dither_buffer_c(int thread_id, int thread_num, int seed);
void generate_dither_buffer_sse2(int thread_id, int thread_num, int seed);
void generate_dither_buffer_sse41(int thread_id, int thread_num, int seed);
void generate_dither_buffer_avx(int thread_id, int thread_num, int seed);

void decrease_banding_mode0_c(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode12_c(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);

void decrease_banding_mode0_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode0_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode0_p_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode0_i_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode0_p_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode0_i_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode0_p_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode0_i_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode0_p_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode0_i_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode0_p_avx2_new(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode0_i_avx2_new(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);

void decrease_banding_mode1_blur_later_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_first_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_later_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_first_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_later_p_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_first_p_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_later_i_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_first_i_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_later_p_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_first_p_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_later_i_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_first_i_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_later_p_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_first_p_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_later_i_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_first_i_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_later_p_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_first_p_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_later_i_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_first_i_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_later_p_avx2_new(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_first_p_avx2_new(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_later_i_avx2_new(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode1_blur_first_i_avx2_new(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);

void decrease_banding_mode2_blur_later_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_first_p_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_later_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_first_i_sse2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_later_p_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_first_p_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_later_i_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_first_i_ssse3(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_later_p_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_first_p_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_later_i_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_first_i_sse41(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_later_p_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_first_p_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_later_i_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_first_i_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_later_p_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_first_p_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_later_i_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);
void decrease_banding_mode2_blur_first_i_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip);

typedef struct {
    func_decrease_banding set[2][2];
} func_decrease_banding_mode_t;

enum {
    NONE   = 0x0000,
    SSE2   = 0x0001,
    SSE3   = 0x0002,
    SSSE3  = 0x0004,
    SSE41  = 0x0008,
    SSE42  = 0x0010,
    POPCNT = 0x0020,
    AVX    = 0x0040,
    AVX2   = 0x0080,
};

typedef struct {
    union {
        xor514_t   *gen_rand;
        xor514x2_t *gen_rand_avx2;
    };
    int thread_num;         //現在gen_randで確保している数
    int current_thread_num; //直近のスレッド数
    int block_count_x;      //ブロック分割数(横)
    int block_count_y;      //ブロック分割数(縦)
    int _seed;              //現在の設定(要チェック)
    DWORD availableSIMD;
    func_decrease_banding_mode_t decrease_banding[3]; //sample_mode別のバンディング低減関数
} banding_t;

EXTERN banding_t band;

void band_get_block_range(int ib, int width, int height, int *x_start, int *x_fin, int *y_start, int *y_fin);

DWORD get_availableSIMD();

void band_set_func(); //generate_dither_buffer, decrease_banding[]の関数を設定

#endif //_BANDING_H_
