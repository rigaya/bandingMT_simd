//  -----------------------------------------------------------------------------------------
//    バンディング低減MT SIMD by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------


#include <Windows.h>
#include <emmintrin.h> //SSE2
#include <smmintrin.h> //SSE4.1
#include "banding.h"
#include "xor_rand.h"
#include "filter.h"
#include "banding_simd.h"

#if _MSC_VER >= 1800 && !defined(__AVX__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX or /arch:AVX2 for this file.");
#endif

void decrease_banding_mode0_p_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode0_simd(thread_id, thread_num, fp, fpip, FALSE, AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode0_i_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode0_simd(thread_id, thread_num, fp, fpip, TRUE, AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode1_blur_first_p_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, TRUE, FALSE, AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode1_blur_first_i_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, TRUE, TRUE, AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode1_blur_later_p_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, FALSE, FALSE, AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode1_blur_later_i_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode1_simd(thread_id, thread_num, fp, fpip, FALSE, TRUE, AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode2_blur_first_p_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, TRUE, FALSE, AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode2_blur_first_i_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, TRUE, TRUE, AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode2_blur_later_p_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, FALSE, FALSE, AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode2_blur_later_i_avx(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    decrease_banding_mode2_simd(thread_id, thread_num, fp, fpip, FALSE, TRUE, AVX|SSE41|SSSE3|SSE2);
}
