//  -----------------------------------------------------------------------------------------
//    バンディング低減MT SIMD by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

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
