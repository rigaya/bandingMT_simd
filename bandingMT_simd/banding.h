//  -----------------------------------------------------------------------------------------
//    バンディング低減MT SIMD by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------


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
    int _seed;              ///現在の設定(要チェック)
    DWORD availableSIMD;
    func_decrease_banding_mode_t decrease_banding[3]; //sample_mode別のバンディング低減関数
} banding_t;

EXTERN banding_t band;

DWORD get_availableSIMD();

void band_set_func(); //generate_dither_buffer, decrease_banding[]の関数を設定

#endif //_BANDING_H_
