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
#include <immintrin.h> //AVX, AVX2
#include "banding.h"
#include "xor_rand.h"
#include "filter.h"
#include "banding_simd.h"

#define USE_VPGATHER 0

//mode012共通 ... ref用乱数の見を発生させる
static void __forceinline createRandAVX2_0(BYTE *ref_ptr, xor514x2_t *gen_rand, __m256i yRangeYLimit, __m256i& yRangeXLimit0, __m256i& yRangeXLimit1) {
	__m256i y0, y1;
	__m128i x0, x1;
	__m256i yRange = _mm256_min_epu16(yRangeYLimit, _mm256_min_epu16(yRangeXLimit0, yRangeXLimit1));
	__m256i yRange2 = _mm256_adds_epu16(_mm256_slli_epi16(yRange, 1), yOne256);

	xor514x2(gen_rand);
	x0 = gen_rand->m[6];
	x1 = gen_rand->m[7];
	y0 = _mm256_cvtepu8_epi16(x0);
	y1 = _mm256_cvtepu8_epi16(_mm_xor_si128(x0, x1));
	y0 = _mm256_mullo_epi16(y0, yRange2);
	y1 = _mm256_mullo_epi16(y1, yRange2);
	y0 = _mm256_srai_epi16(y0, 8);
	y1 = _mm256_srai_epi16(y1, 8);
	y0 = _mm256_or_si256(y0, _mm256_slli_si256(y1, 1));
	y0 = _mm256_sub_epi8(y0, _mm256_or_si256(yRange, _mm256_slli_si256(yRange, 1)));
	_mm256_store_si256((__m256i*)(ref_ptr), y0);
	
	yRangeXLimit0 = _mm256_adds_epu16(yRangeXLimit0, yC_16);
	yRangeXLimit1 = _mm256_subs_epu16(yRangeXLimit1, yC_16);
}
//mode12用 ... dither用乱数のうち2つを発生させる、最後の一つは次で
static void __forceinline createRandAVX2_1(short *dither_ptr, xor514x2_t *gen_rand, const short *ditherYC2, const short *ditherYC) {
	__m256i y0, y1;
	xor514x2(gen_rand);
	y0 = _mm256_cvtepu8_epi16(gen_rand->m[6]);
	y1 = _mm256_cvtepu8_epi16(gen_rand->m[7]);
	y0 = _mm256_mullo_epi16(y0, _mm256_load_si256((__m256i*)(ditherYC2 +  0)));
	y1 = _mm256_mullo_epi16(y1, _mm256_load_si256((__m256i*)(ditherYC2 + 16)));
	y0 = _mm256_srai_epi16(y0, 8);
	y1 = _mm256_srai_epi16(y1, 8);
	y0 = _mm256_sub_epi16(y0, _mm256_load_si256((__m256i*)(ditherYC +  0)));
	y1 = _mm256_sub_epi16(y1, _mm256_load_si256((__m256i*)(ditherYC + 16)));
	_mm256_store_si256((__m256i*)(dither_ptr +  0), y0);
	_mm256_store_si256((__m256i*)(dither_ptr + 16), y1);
}
//mode12用 ... 次で使うref用乱数とdither用乱数の最後の一つを発生させる
static void __forceinline createRandAVX2_2(short *dither_ptr, BYTE *ref_ptr, xor514x2_t *gen_rand, const short *ditherYC2, const short *ditherYC, __m256i yRangeYLimit, __m256i& yRangeXLimit0, __m256i& yRangeXLimit1) {
	__m256i y0, y1;
	__m128i x0, x1;
	__m256i yRange = _mm256_min_epu16(yRangeYLimit, _mm256_min_epu16(yRangeXLimit0, yRangeXLimit1));
	__m256i yRange2 = _mm256_adds_epu16(_mm256_slli_epi16(yRange, 1), yOne256);

	xor514x2(gen_rand);
	x1 = gen_rand->m[7];
	y0 = _mm256_cvtepu8_epi16(x1);
	y0 = _mm256_mullo_epi16(y0, _mm256_load_si256((__m256i*)(ditherYC2 + 32)));
	y0 = _mm256_srai_epi16(y0, 8);
	y0 = _mm256_sub_epi16(y0, _mm256_load_si256((__m256i*)(ditherYC + 32)));
	_mm256_store_si256((__m256i*)(dither_ptr + 32), y0);
	
	x0 = gen_rand->m[6];
	y0 = _mm256_cvtepu8_epi16(x0);
	y1 = _mm256_cvtepu8_epi16(_mm_xor_si128(x0, x1));
	y0 = _mm256_mullo_epi16(y0, yRange2);
	y1 = _mm256_mullo_epi16(y1, yRange2);
	y0 = _mm256_srai_epi16(y0, 8);
	y1 = _mm256_srai_epi16(y1, 8);
	y0 = _mm256_or_si256(y0, _mm256_slli_si256(y1, 1));
	y0 = _mm256_sub_epi8(y0, _mm256_or_si256(yRange, _mm256_slli_si256(yRange, 1)));
	_mm256_store_si256((__m256i*)(ref_ptr), y0);
	
	yRangeXLimit0 = _mm256_adds_epu16(yRangeXLimit0, yC_16);
	yRangeXLimit1 = _mm256_subs_epu16(yRangeXLimit1, yC_16);
}
//mode0用 ... ref用乱数のみを発生させる
static void __forceinline createRandAVX2_3(BYTE *ref_ptr, xor514x2_t *gen_rand, __m256i yRangeYLimit, __m256i& yRangeXLimit0, __m256i& yRangeXLimit1) {
	__m256i y0, y1;
	__m128i x0, x1;
	__m256i yRange = _mm256_min_epu16(yRangeYLimit, _mm256_min_epu16(yRangeXLimit0, yRangeXLimit1));
	__m256i yRange2 = _mm256_adds_epu16(_mm256_slli_epi16(yRange, 1), yOne256);

	xor514x2(gen_rand);
	x0 = gen_rand->m[6];
	x1 = gen_rand->m[7];
	y0 = _mm256_cvtepu8_epi16(x0);
	y1 = _mm256_cvtepu8_epi16(x1);
	y0 = _mm256_mullo_epi16(y0, yRange2);
	y1 = _mm256_mullo_epi16(y1, yRange2);
	y0 = _mm256_srai_epi16(y0, 8);
	y1 = _mm256_srai_epi16(y1, 8);
	y0 = _mm256_or_si256(y0, _mm256_slli_si256(y1, 1));
	y0 = _mm256_sub_epi8(y0, _mm256_or_si256(yRange, _mm256_slli_si256(yRange, 1)));
	_mm256_store_si256((__m256i*)(ref_ptr), y0);
	
	yRangeXLimit0 = _mm256_adds_epu16(yRangeXLimit0, yC_16);
	yRangeXLimit1 = _mm256_subs_epu16(yRangeXLimit1, yC_16);
}
//mode12用 ... dither用乱数のみを発生させy0,y1,y2で返す
static void __forceinline createRandAVX2_4(xor514x2_t *gen_rand, const short *ditherYC2, const short *ditherYC, __m256i& y0, __m256i& y1, __m256i& y2) {
	xor514x2(gen_rand);
	y0 = _mm256_cvtepu8_epi16(gen_rand->m[6]);
	y1 = _mm256_cvtepu8_epi16(gen_rand->m[7]);
	y2 = _mm256_cvtepu8_epi16(_mm_xor_si128(gen_rand->m[6], gen_rand->m[7]));
	y0 = _mm256_mullo_epi16(y0, _mm256_load_si256((__m256i*)(ditherYC2 +  0)));
	y1 = _mm256_mullo_epi16(y1, _mm256_load_si256((__m256i*)(ditherYC2 + 16)));
	y2 = _mm256_mullo_epi16(y2, _mm256_load_si256((__m256i*)(ditherYC2 + 32)));
	y0 = _mm256_srai_epi16(y0, 8);
	y1 = _mm256_srai_epi16(y1, 8);
	y2 = _mm256_srai_epi16(y2, 8);
	y0 = _mm256_sub_epi16(y0, _mm256_load_si256((__m256i*)(ditherYC +  0)));
	y1 = _mm256_sub_epi16(y1, _mm256_load_si256((__m256i*)(ditherYC + 16)));
	y2 = _mm256_sub_epi16(y2, _mm256_load_si256((__m256i*)(ditherYC + 32)));
}

#if USE_VPGATHER
static const char _declspec(align(64)) PACK_YC48_SHUFFLE[96] = {
	10, 11, 12, 13, -1, -1, -1, -1,  0,  1,  2,  3,  4,  5,  8,  9,
	-1, -1, -1, -1,  0,  1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 13,
	 0,  1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 13, -1, -1, -1, -1,
	 4,  5,  8,  9, 10, 11, 12, 13, -1, -1, -1, -1,  0,  1,  2,  3,
	-1, -1, -1, -1,  0,  1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 13,
	-1, -1, -1, -1,  0,  1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 13,
};

static void __forceinline pack_yc48_1(__m256i& yGather0, __m256i& yGather1) {
	__m256i yTemp0, yTemp1, yTemp2;
	yTemp0   = _mm256_bsrli_epi128(yGather0, 2);
	yGather0 = _mm256_blend_epi16(yGather0, yTemp0, 0x20+0x10+0x08);
	yGather1 = _mm256_shuffle_epi8(yGather1, _mm256_load_si256((__m256i*)PACK_YC48_SHUFFLE));
	yTemp0   = _mm256_blend_epi32(yGather1, yGather0, 0x10);
	yTemp0   = _mm256_permute4x64_epi64(yTemp0, _MM_SHUFFLE(1,0,3,2));
	yTemp1   = _mm256_bsrli_epi128(yGather0, 4);
	yTemp1   = _mm256_blend_epi32(yTemp1, yTemp0, 0x80+0x40);
	yTemp2   = _mm256_bslli_epi128(yTemp0, 12);
	yGather0 = _mm256_blend_epi32(yGather0, yTemp1, 0xf0);
	yGather0 = _mm256_blend_epi32(yGather0, yTemp2, 0x08);
	yGather1 = _mm256_blend_epi32(yGather1, yTemp0, 0x08+0x04+0x02);
}
static void __forceinline pack_yc48_2(__m256i& yGather1, __m256i& yGather2, __m256i yGather3) {
	__m256i yTemp0, yTemp1, yTemp2;
	yGather2 = _mm256_shuffle_epi8(yGather2, _mm256_load_si256((__m256i*)(PACK_YC48_SHUFFLE + 32)));
	yGather3 = _mm256_shuffle_epi8(yGather3, _mm256_load_si256((__m256i*)(PACK_YC48_SHUFFLE + 64)));
	yTemp0   = _mm256_blend_epi32(yGather2, yGather3, 0x80+0x40+0x08);
	yTemp0   = _mm256_permute4x64_epi64(yTemp0, _MM_SHUFFLE(1,0,3,2));
	yTemp2   = _mm256_bslli_epi128(yGather3, 4);
	yGather1 = _mm256_blend_epi32(yGather1, yGather2, 0x80);
	yGather1 = _mm256_blend_epi32(yGather1, yTemp0, 0x40+0x20+0x10);
	yTemp1   = _mm256_bsrli_epi128(yTemp0, 12);
	yGather2 = _mm256_blend_epi32(yGather3, yTemp2, 0x0f);
	yTemp0   = _mm256_blend_epi32(yTemp0, yTemp1, 0xf0+0x08+0x04);
	yGather2 = _mm256_or_si256(yGather2, yTemp0);
}
static void __forceinline decrease_banding_mode0_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip, BOOL process_per_field, DWORD simd) {
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
	const int  y_start         = ( height *  thread_id    ) / thread_num;
	const int  y_end           = ( height * (thread_id+1) ) / thread_num;
	__m256i yRefMulti = _mm256_unpacklo_epi16(_mm256_set1_epi16(max_w), yOne256);
	__m256i yGather0, yGather1, yGather2, yGather3;
	
	BYTE  __declspec(align(32)) ref[32];
	short __declspec(align(32)) threshold[48];

	for (int i = 0; i < 16; i++) {
		threshold[3*i+0] = threshold_y;
		threshold[3*i+1] = threshold_cb;
		threshold[3*i+2] = threshold_cr;
	}
	xor514x2_t gen_rand;
	if (!rand_each_frame) {
		xor514x2_init(&gen_rand, seed + (y_start << 2));
	} else {
		gen_rand = band.gen_rand_avx2[thread_id];
	}
	
	int y;
	const int y_main_end = min(y_end, height - 1);
	for (y = y_start; y < y_main_end; y++) {
		PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w;
		PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w;
		__m256i yRangeYLimit = _mm256_set1_epi16(min3(range, y, height - y - 1));
		__m256i yRangeXLimit0 = _mm256_load_si256((__m256i*)x_range_offset);
		__m256i yRangeXLimit1 = _mm256_subs_epu16(_mm256_set1_epi16(width - 1), _mm256_load_si256((__m256i*)x_range_offset));
		createRandAVX2_0(ref, &gen_rand, yRangeYLimit, yRangeXLimit0, yRangeXLimit1);
		for (int i_step = 0, x = width - 16; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {
			__m256i yRef = _mm256_loadu_si256((__m256i*)ref);
			if (process_per_field) {
				__m256i yFeildMask = _mm256_slli_epi16(_mm256_cmpeq_epi8(_mm256_setzero_si256(), _mm256_setzero_si256()), 1);
				yRef = _mm256_and_si256(yRef, yFeildMask);
			}
			__m256i yRefUpper = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(yRef, 1));
			__m256i yRefLower = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(yRef));
			
			yRefLower = _mm256_multi3_epi32(_mm256_add_epi32(_mm256_madd_epi16(yRefLower, yRefMulti), _mm256_load_si256((__m256i*)&ref_offset[0])));
			yRefUpper = _mm256_multi3_epi32(_mm256_add_epi32(_mm256_madd_epi16(yRefUpper, yRefMulti), _mm256_load_si256((__m256i*)&ref_offset[8])));
			
			yGather1 = _mm256_i32gather_epi64((__int64 *)ycp_src, _mm256_extracti128_si256(yRefLower, 1), 2);
			yGather0 = _mm256_i32gather_epi64((__int64 *)ycp_src, _mm256_castsi256_si128(yRefLower), 2);
			pack_yc48_1(yGather0, yGather1);

			__m256i yYCPRef0, yYCPDiff, yYCP, yThreshold, yBase, yMask;
			
			yYCPRef0 = yGather0;
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src +  0));
			yYCPDiff = _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0));
			yThreshold = _mm256_load_si256((__m256i*)(threshold +  0));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, yYCPRef0, yMask);  //r = (mask0 & 0xff) ? b : a
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst +  0), yBase);
			
			yGather3 = _mm256_i32gather_epi64((__int64 *)ycp_src, _mm256_extracti128_si256(yRefUpper, 1), 2);
			yGather2 = _mm256_i32gather_epi64((__int64 *)ycp_src, _mm256_castsi256_si128(yRefUpper), 2);
			pack_yc48_2(yGather1, yGather2, yGather3);
			createRandAVX2_3(ref, &gen_rand, yRangeYLimit, yRangeXLimit0, yRangeXLimit1);
		
			yYCPRef0 = yGather1;
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 32));
			yYCPDiff = _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0));
			yThreshold = _mm256_load_si256((__m256i*)(threshold +  16));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, yYCPRef0, yMask);  //r = (mask0 & 0xff) ? b : a
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 32), yBase);

			yYCPRef0 = yGather2;
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 64));
			yYCPDiff = _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0));
			yThreshold = _mm256_load_si256((__m256i*)(threshold + 32));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, yYCPRef0, yMask);  //r = (mask0 & 0xff) ? b : a
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 64), yBase);

			i_step = limit_1_to_16(x);
		}
	}
	//最後のライン
	if (y < y_end)
		avx2_memcpy((BYTE *)(fpip->ycp_temp + y * max_w), (BYTE *)(fpip->ycp_edit + y * max_w), width * 6);
	_mm256_zeroupper();
	band.gen_rand_avx2[thread_id] = gen_rand;
}
#else
static void __forceinline decrease_banding_mode0_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip, BOOL process_per_field, DWORD simd) {
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
	const int  y_start         = ( height *  thread_id    ) / thread_num;
	const int  y_end           = ( height * (thread_id+1) ) / thread_num;
	__m256i yRefMulti = _mm256_unpacklo_epi16(_mm256_set1_epi16(max_w), yOne256);
	
	BYTE     __declspec(align(32)) ref[32];
	PIXEL_YC __declspec(align(32)) ycp_buffer[16];
	int      __declspec(align(32)) ref_buffer[16];
	short    __declspec(align(32)) threshold[48];

	for (int i = 0; i < 16; i++) {
		threshold[3*i+0] = threshold_y;
		threshold[3*i+1] = threshold_cb;
		threshold[3*i+2] = threshold_cr;
	}
	xor514x2_t gen_rand;
	if (!rand_each_frame) {
		xor514x2_init(&gen_rand, seed + (y_start << 2));
	} else {
		gen_rand = band.gen_rand_avx2[thread_id];
	}
	
	int y;
	const int y_main_end = min(y_end, height - 1);
	for (y = y_start; y < y_main_end; y++) {
		PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w;
		PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w;
		__m256i yRangeYLimit = _mm256_set1_epi16(min3(range, y, height - y - 1));
		__m256i yRangeXLimit0 = _mm256_load_si256((__m256i*)x_range_offset);
		__m256i yRangeXLimit1 = _mm256_subs_epu16(_mm256_set1_epi16(width - 1), _mm256_load_si256((__m256i*)x_range_offset));
		createRandAVX2_0(ref, &gen_rand, yRangeYLimit, yRangeXLimit0, yRangeXLimit1);
		for (int i_step = 0, x = width - 16; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {
			__m256i yRef = _mm256_load_si256((__m256i*)ref);
			if (process_per_field)
				yRef = apply_field_mask_256(yRef, TRUE);
			__m256i yRefUpper = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(yRef, 1));
			__m256i yRefLower = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(yRef));
			
			yRefLower = _mm256_add_epi32(_mm256_madd_epi16(yRefLower, yRefMulti), _mm256_load_si256((__m256i*)&ref_offset[0]));
			yRefUpper = _mm256_add_epi32(_mm256_madd_epi16(yRefUpper, yRefMulti), _mm256_load_si256((__m256i*)&ref_offset[8]));

			_mm256_store_si256((__m256i*)&ref_buffer[0], _mm256_multi6_epi32(yRefLower));
			_mm256_store_si256((__m256i*)&ref_buffer[8], _mm256_multi6_epi32(yRefUpper));
			
			{
				__m64 m0, m1, m2, m3, m4, m5, m6, m7;
				m0 = *(__m64*)((BYTE *)ycp_src + ref_buffer[0]);
				m1 = *(__m64*)((BYTE *)ycp_src + ref_buffer[1]);
				m2 = *(__m64*)((BYTE *)ycp_src + ref_buffer[2]);
				m3 = *(__m64*)((BYTE *)ycp_src + ref_buffer[3]);
				m4 = *(__m64*)((BYTE *)ycp_src + ref_buffer[4]);
				m5 = *(__m64*)((BYTE *)ycp_src + ref_buffer[5]);
				m6 = *(__m64*)((BYTE *)ycp_src + ref_buffer[6]);
				m7 = *(__m64*)((BYTE *)ycp_src + ref_buffer[7]);
				m2 = _mm_shuffle_pi16(m2, _MM_SHUFFLE(2,2,1,0));
				m6 = _mm_shuffle_pi16(m6, _MM_SHUFFLE(2,2,1,0));
				m3 = _mm_alignr_pi8(m3, m2, 6);
				m7 = _mm_alignr_pi8(m7, m6, 6);
				*(__m64 *)((BYTE *)ycp_buffer + 16) = m3;
				*(__m64 *)((BYTE *)ycp_buffer + 40) = m7;
				m1 = _mm_shuffle_pi16(m1, _MM_SHUFFLE(2,1,1,0));
				m5 = _mm_shuffle_pi16(m5, _MM_SHUFFLE(2,1,1,0));
				m2 = _mm_alignr_pi8(m2, m1, 4);
				m6 = _mm_alignr_pi8(m6, m5, 4);
				*(__m64 *)((BYTE *)ycp_buffer +  8) = m2;
				*(__m64 *)((BYTE *)ycp_buffer + 32) = m6;
				m0 = _mm_shuffle_pi16(m0, _MM_SHUFFLE(2,1,0,0));
				m4 = _mm_shuffle_pi16(m4, _MM_SHUFFLE(2,1,0,0));
				m1 = _mm_alignr_pi8(m1, m0, 2);
				m5 = _mm_alignr_pi8(m5, m4, 2);
				*(__m64 *)((BYTE *)ycp_buffer +  0) = m1;
				*(__m64 *)((BYTE *)ycp_buffer + 24) = m5;
			}

			__m256i yYCPRef0, yYCPDiff, yYCP, yThreshold, yBase, yMask;

			yYCPRef0 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer +   0));
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src +  0));
			yYCPDiff = _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0));
			yThreshold = _mm256_load_si256((__m256i*)(threshold +  0));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, yYCPRef0, yMask);  //r = (mask0 & 0xff) ? b : a
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst +  0), yBase);
		
			{
				__m64 m0, m1, m2, m3, m4, m5, m6, m7;
				m0 = *(__m64*)((BYTE *)ycp_src + ref_buffer[ 8]);
				m1 = *(__m64*)((BYTE *)ycp_src + ref_buffer[ 9]);
				m2 = *(__m64*)((BYTE *)ycp_src + ref_buffer[10]);
				m3 = *(__m64*)((BYTE *)ycp_src + ref_buffer[11]);
				m4 = *(__m64*)((BYTE *)ycp_src + ref_buffer[12]);
				m5 = *(__m64*)((BYTE *)ycp_src + ref_buffer[13]);
				m6 = *(__m64*)((BYTE *)ycp_src + ref_buffer[14]);
				m7 = *(__m64*)((BYTE *)ycp_src + ref_buffer[15]);
				m2 = _mm_shuffle_pi16(m2, _MM_SHUFFLE(2,2,1,0));
				m6 = _mm_shuffle_pi16(m6, _MM_SHUFFLE(2,2,1,0));
				m3 = _mm_alignr_pi8(m3, m2, 6);
				m7 = _mm_alignr_pi8(m7, m6, 6);
				*(__m64 *)((BYTE *)ycp_buffer + 64) = m3;
				*(__m64 *)((BYTE *)ycp_buffer + 88) = m7;
				m1 = _mm_shuffle_pi16(m1, _MM_SHUFFLE(2,1,1,0));
				m5 = _mm_shuffle_pi16(m5, _MM_SHUFFLE(2,1,1,0));
				m2 = _mm_alignr_pi8(m2, m1, 4);
				m6 = _mm_alignr_pi8(m6, m5, 4);
				*(__m64 *)((BYTE *)ycp_buffer + 56) = m2;
				*(__m64 *)((BYTE *)ycp_buffer + 80) = m6;
				m0 = _mm_shuffle_pi16(m0, _MM_SHUFFLE(2,1,0,0));
				m4 = _mm_shuffle_pi16(m4, _MM_SHUFFLE(2,1,0,0));
				m1 = _mm_alignr_pi8(m1, m0, 2);
				m5 = _mm_alignr_pi8(m5, m4, 2);
				*(__m64 *)((BYTE *)ycp_buffer + 48) = m1;
				*(__m64 *)((BYTE *)ycp_buffer + 72) = m5;
			}
			createRandAVX2_3(ref, &gen_rand, yRangeYLimit, yRangeXLimit0, yRangeXLimit1);
		
			yYCPRef0 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer + 32));
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 32));
			yYCPDiff = _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0));
			yThreshold = _mm256_load_si256((__m256i*)(threshold +  16));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, yYCPRef0, yMask);  //r = (mask0 & 0xff) ? b : a
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 32), yBase);


			yYCPRef0 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer + 64));
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 64));
			yYCPDiff = _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0));
			yThreshold = _mm256_load_si256((__m256i*)(threshold + 32));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, yYCPRef0, yMask);  //r = (mask0 & 0xff) ? b : a
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 64), yBase);

			i_step = limit_1_to_16(x);
		}
	}
	//最後のライン
	if (y < y_end)
		avx2_memcpy((BYTE *)(fpip->ycp_temp + y * max_w), (BYTE *)(fpip->ycp_edit + y * max_w), width * 6);
	_mm256_zeroupper();
	_mm_empty();
	band.gen_rand_avx2[thread_id] = gen_rand;
}
#endif
void decrease_banding_mode0_p_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
	decrease_banding_mode0_avx2(thread_id, thread_num, fp, fpip, FALSE, AVX2|AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode0_i_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
	decrease_banding_mode0_avx2(thread_id, thread_num, fp, fpip, TRUE, AVX2|AVX|SSE41|SSSE3|SSE2);
}


#pragma warning (push)
#pragma warning (disable: 4799) //warning C4799: emms命令がありません
static void __forceinline gather_ycp(PIXEL_YC *ycp_buffer, const PIXEL_YC *ycp_src, const int *ref_buffer, int i) {
	__m64 m0, m1, m2, m3, m4, m5, m6, m7;
	m0 = *(__m64*)((BYTE *)ycp_src + ref_buffer[i+0]);
	m1 = *(__m64*)((BYTE *)ycp_src + ref_buffer[i+1]);
	m2 = *(__m64*)((BYTE *)ycp_src + ref_buffer[i+2]);
	m3 = *(__m64*)((BYTE *)ycp_src + ref_buffer[i+3]);
	m4 = *(__m64*)((BYTE *)ycp_src - ref_buffer[i+0] + i*12 + 0);
	m5 = *(__m64*)((BYTE *)ycp_src - ref_buffer[i+1] + i*12 + 12);
	m6 = *(__m64*)((BYTE *)ycp_src - ref_buffer[i+2] + i*12 + 24);
	m7 = *(__m64*)((BYTE *)ycp_src - ref_buffer[i+3] + i*12 + 36);
	m2 = _mm_shuffle_pi16(m2, _MM_SHUFFLE(2,2,1,0));
	m6 = _mm_shuffle_pi16(m6, _MM_SHUFFLE(2,2,1,0));
	m3 = _mm_alignr_pi8(m3, m2, 6);
	m7 = _mm_alignr_pi8(m7, m6, 6);
	*(__m64 *)((BYTE *)ycp_buffer +i*6+  16) = m3;
	*(__m64 *)((BYTE *)ycp_buffer +i*6+ 112) = m7;
	m1 = _mm_shuffle_pi16(m1, _MM_SHUFFLE(2,1,1,0));
	m5 = _mm_shuffle_pi16(m5, _MM_SHUFFLE(2,1,1,0));
	m2 = _mm_alignr_pi8(m2, m1, 4);
	m6 = _mm_alignr_pi8(m6, m5, 4);
	*(__m64 *)((BYTE *)ycp_buffer +i*6+   8) = m2;
	*(__m64 *)((BYTE *)ycp_buffer +i*6+ 104) = m6;
	m0 = _mm_shuffle_pi16(m0, _MM_SHUFFLE(2,1,0,0));
	m4 = _mm_shuffle_pi16(m4, _MM_SHUFFLE(2,1,0,0));
	m1 = _mm_alignr_pi8(m1, m0, 2);
	m5 = _mm_alignr_pi8(m5, m4, 2);
	*(__m64 *)((BYTE *)ycp_buffer +i*6+   0) = m1;
	*(__m64 *)((BYTE *)ycp_buffer +i*6+  96) = m5;
}
#pragma warning (pop)

#if USE_VPGATHER
//blur_first、process_per_field、simdは定数として与え、
//条件分岐をコンパイル時に削除させる
static void __forceinline decrease_banding_mode1_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip, BOOL blur_first,  BOOL process_per_field, DWORD simd) {
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
	const int  y_start         = ( height *  thread_id    ) / thread_num;
	const int  y_end           = ( height * (thread_id+1) ) / thread_num;
	__m256i yRefMulti = _mm256_unpacklo_epi16(_mm256_set1_epi16(max_w), yOne256);
	__m256i yGather0, yGather1, yGather2, yGather3, yGather4, yGather5, yGather6, yGather7;
	
	BYTE  __declspec(align(32)) ref[32];
	short __declspec(align(32)) dither[48];
	short __declspec(align(32)) ditherYC[48];
	short __declspec(align(32)) ditherYC2[48];
	short __declspec(align(32)) threshold[48];

	for (int i = 0; i < 16; i++) {
		threshold[3*i+0] = threshold_y;
		threshold[3*i+1] = threshold_cb;
		threshold[3*i+2] = threshold_cr;
	}
	for (int i = 0; i < 16; i++) {
		ditherYC[3*i+0] = ditherY;
		ditherYC[3*i+1] = ditherC;
		ditherYC[3*i+2] = ditherC;
	}
	{
		__m256i y0 = _mm256_load_si256((__m256i *)(ditherYC +  0));
		__m256i y1 = _mm256_load_si256((__m256i *)(ditherYC + 16));
		__m256i y2 = _mm256_load_si256((__m256i *)(ditherYC + 32));
		y0 = _mm256_slli_epi16(y0, 1);
		y1 = _mm256_slli_epi16(y1, 1);
		y2 = _mm256_slli_epi16(y2, 1);
		y0 = _mm256_add_epi16(y0, yOne256);
		y1 = _mm256_add_epi16(y1, yOne256);
		y2 = _mm256_add_epi16(y2, yOne256);
		_mm256_store_si256((__m256i *)(ditherYC2 +  0), y0);
		_mm256_store_si256((__m256i *)(ditherYC2 + 16), y1);
		_mm256_store_si256((__m256i *)(ditherYC2 + 32), y2);
	}
	xor514x2_t gen_rand;
	if (!rand_each_frame) {
		xor514x2_init(&gen_rand, seed + (y_start << 2));
	} else {
		gen_rand = band.gen_rand_avx2[thread_id];
	}

	int y;
	const int y_main_end = min(y_end, height - 1);
	for (y = y_start; y < y_main_end; y++) {
		PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w;
		PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w;
		__m256i yRangeYLimit = _mm256_set1_epi16(min3(range, y, height - y - 1));
		__m256i yRangeXLimit0 = _mm256_load_si256((__m256i*)x_range_offset);
		__m256i yRangeXLimit1 = _mm256_subs_epu16(_mm256_set1_epi16(width - 1), _mm256_load_si256((__m256i*)x_range_offset));
		createRandAVX2_0(ref, &gen_rand, yRangeYLimit, yRangeXLimit0, yRangeXLimit1);
		for (int i_step = 0, x = width - 16; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {
			__m256i yRef = _mm256_loadu_si256((__m256i*)ref);
			if (process_per_field) {
				yRef = apply_field_mask_256(yRef, TRUE);
			}
			__m256i yRefUpper = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(yRef, 1));
			__m256i yRefLower = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(yRef));

			yRefLower = _mm256_madd_epi16(yRefLower, yRefMulti);
			yRefUpper = _mm256_madd_epi16(yRefUpper, yRefMulti);
			
			__m256i yRefLowerP = _mm256_multi3_epi32(_mm256_add_epi32(yRefLower, _mm256_load_si256((__m256i*)&ref_offset[0])));
			__m256i yRefLowerM = _mm256_multi3_epi32(_mm256_sub_epi32(yRefLower, _mm256_load_si256((__m256i*)&ref_offset[0])));

			yRefLowerM = _mm256_neg_epi32(yRefLowerM);
			
			yGather1 = _mm256_i32gather_epi64((__int64 *)ycp_src, _mm256_extracti128_si256(yRefLowerP, 1), 2);
			yGather0 = _mm256_i32gather_epi64((__int64 *)ycp_src, _mm256_castsi256_si128(yRefLowerP), 2);
			yGather5 = _mm256_i32gather_epi64((__int64 *)ycp_src, _mm256_extracti128_si256(yRefLowerM, 1), 2);
			yGather4 = _mm256_i32gather_epi64((__int64 *)ycp_src, _mm256_castsi256_si128(yRefLowerM), 2);

			pack_yc48_1(yGather0, yGather1);
			pack_yc48_1(yGather4, yGather5);
			createRandAVX2_1(dither, &gen_rand, ditherYC2, ditherYC);

			__m256i yYCPRef0, yYCPRef1, xYCPAvg, yYCPDiff, yYCP, yThreshold, yBase, yMask, yDither;

			yYCPRef0 = yGather0;
			yYCPRef1 = yGather4;
			xYCPAvg  = _mm256_srai_epi16(_mm256_adds_epi16(_mm256_adds_epi16(yYCPRef0, yYCPRef1), yOne256), 1);
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src +  0));
			yYCPDiff = (blur_first) ? _mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPAvg))
									: _mm256_max_epi16(_mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0)),
									                   _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef1)) );
			yThreshold = _mm256_load_si256((__m256i*)(threshold +  0));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, xYCPAvg, yMask);  //r = (mask0 & 0xff) ? b : a
			yDither  = _mm256_load_si256((__m256i *)(dither +  0));
			yYCP     = _mm256_adds_epi16(yBase, yDither);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst +  0), yYCP);

			__m256i yRefUpperP = _mm256_multi3_epi32(_mm256_add_epi32(yRefUpper, _mm256_load_si256((__m256i*)&ref_offset[8])));
			__m256i yRefUpperM = _mm256_multi3_epi32(_mm256_sub_epi32(yRefUpper, _mm256_load_si256((__m256i*)&ref_offset[8])));
			yRefUpperM = _mm256_neg_epi32(yRefUpperM);

			yGather3 = _mm256_i32gather_epi64((__int64 *)ycp_src, _mm256_extracti128_si256(yRefUpperP, 1), 2);
			yGather2 = _mm256_i32gather_epi64((__int64 *)ycp_src, _mm256_castsi256_si128(yRefUpperP), 2);
			yGather7 = _mm256_i32gather_epi64((__int64 *)ycp_src, _mm256_extracti128_si256(yRefUpperM, 1), 2);
			yGather6 = _mm256_i32gather_epi64((__int64 *)ycp_src, _mm256_castsi256_si128(yRefUpperM), 2);
			
			pack_yc48_2(yGather1, yGather2, yGather3);
			pack_yc48_2(yGather5, yGather6, yGather7);
			createRandAVX2_2(dither, ref, &gen_rand, ditherYC2, ditherYC, yRangeYLimit, yRangeXLimit0, yRangeXLimit1);

			yYCPRef0 = yGather1;
			yYCPRef1 = yGather5;
			xYCPAvg  = _mm256_srai_epi16(_mm256_adds_epi16(_mm256_adds_epi16(yYCPRef0, yYCPRef1), yOne256), 1);
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 32));
			yYCPDiff = (blur_first) ? _mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPAvg))
									: _mm256_max_epi16(_mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0)),
									                   _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef1)) );
			yThreshold = _mm256_load_si256((__m256i*)(threshold + 16));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, xYCPAvg, yMask);  //r = (mask0 & 0xff) ? b : a
			yDither  = _mm256_load_si256((__m256i *)(dither + 16));
			yYCP     = _mm256_adds_epi16(yBase, yDither);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 32), yYCP);
			
			

			yYCPRef0 = yGather2;
			yYCPRef1 = yGather6;
			xYCPAvg  = _mm256_srai_epi16(_mm256_adds_epi16(_mm256_adds_epi16(yYCPRef0, yYCPRef1), yOne256), 1);
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 64));
			yYCPDiff = (blur_first) ? _mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPAvg))
									: _mm256_max_epi16(_mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0)),
									                   _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef1)) );
			yThreshold = _mm256_load_si256((__m256i*)(threshold + 32));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, xYCPAvg, yMask);  //r = (mask0 & 0xff) ? b : a
			yDither  = _mm256_load_si256((__m256i *)(dither + 32));
			yYCP     = _mm256_adds_epi16(yBase, yDither);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 64), yYCP);

			i_step = limit_1_to_16(x);
		}
	}
	//最後のライン
	if (y < y_end) {
		PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w;
		PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w;
		for (int i_step = 0, x = width - 16; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {

			__m256i yYCP0, yYCP1, yYCP2, yDither0, yDither1, yDither2;
			createRandAVX2_4(&gen_rand, ditherYC2, ditherYC, yDither0, yDither1, yDither2);

			yYCP0    = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src +  0));
			yYCP1    = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 32));
			yYCP2    = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 64));
			yYCP0    = _mm256_adds_epi16(yYCP0, yDither0);
			yYCP1    = _mm256_adds_epi16(yYCP1, yDither1);
			yYCP2    = _mm256_adds_epi16(yYCP2, yDither2);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst +  0), yYCP0);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 32), yYCP1);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 64), yYCP2);

			i_step = limit_1_to_16(x);
		}
	}
	_mm256_zeroupper();
	band.gen_rand_avx2[thread_id] = gen_rand;
}
#else
static void __forceinline decrease_banding_mode1_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip, BOOL blur_first,  BOOL process_per_field, DWORD simd) {
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
	const int  y_start         = ( height *  thread_id    ) / thread_num;
	const int  y_end           = ( height * (thread_id+1) ) / thread_num;
	__m256i yRefMulti = _mm256_unpacklo_epi16(_mm256_set1_epi16(max_w), yOne256);
	
	BYTE     __declspec(align(32)) ref[32];
	short    __declspec(align(32)) dither[48];
	short    __declspec(align(32)) ditherYC[48];
	short    __declspec(align(32)) ditherYC2[48];
	PIXEL_YC __declspec(align(32)) ycp_buffer[32];
	int      __declspec(align(32)) ref_buffer[16];
	short    __declspec(align(32)) threshold[48];

	for (int i = 0; i < 16; i++) {
		threshold[3*i+0] = threshold_y;
		threshold[3*i+1] = threshold_cb;
		threshold[3*i+2] = threshold_cr;
	}
	for (int i = 0; i < 16; i++) {
		ditherYC[3*i+0] = ditherY;
		ditherYC[3*i+1] = ditherC;
		ditherYC[3*i+2] = ditherC;
	}
	{
		__m256i y0 = _mm256_load_si256((__m256i *)(ditherYC +  0));
		__m256i y1 = _mm256_load_si256((__m256i *)(ditherYC + 16));
		__m256i y2 = _mm256_load_si256((__m256i *)(ditherYC + 32));
		y0 = _mm256_slli_epi16(y0, 1);
		y1 = _mm256_slli_epi16(y1, 1);
		y2 = _mm256_slli_epi16(y2, 1);
		y0 = _mm256_add_epi16(y0, yOne256);
		y1 = _mm256_add_epi16(y1, yOne256);
		y2 = _mm256_add_epi16(y2, yOne256);
		_mm256_store_si256((__m256i *)(ditherYC2 +  0), y0);
		_mm256_store_si256((__m256i *)(ditherYC2 + 16), y1);
		_mm256_store_si256((__m256i *)(ditherYC2 + 32), y2);
	}
	xor514x2_t gen_rand;
	if (!rand_each_frame) {
		xor514x2_init(&gen_rand, seed + (y_start << 2));
	} else {
		gen_rand = band.gen_rand_avx2[thread_id];
	}

	int y;
	const int y_main_end = min(y_end, height - 1);
	for (y = y_start; y < y_main_end; y++) {
		PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w;
		PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w;
		__m256i yRangeYLimit = _mm256_set1_epi16(min3(range, y, height - y - 1));
		__m256i yRangeXLimit0 = _mm256_load_si256((__m256i*)x_range_offset);
		__m256i yRangeXLimit1 = _mm256_subs_epu16(_mm256_set1_epi16(width - 1), _mm256_load_si256((__m256i*)x_range_offset));
		createRandAVX2_0(ref, &gen_rand, yRangeYLimit, yRangeXLimit0, yRangeXLimit1);
		for (int i_step = 0, x = width - 16; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {
			__m256i yRef = _mm256_load_si256((__m256i*)ref);
			if (process_per_field) {
				yRef = apply_field_mask_256(yRef, TRUE);
			}
			__m256i yRefUpper = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(yRef, 1));
			__m256i yRefLower = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(yRef));
			
			yRefLower = _mm256_add_epi32(_mm256_madd_epi16(yRefLower, yRefMulti), _mm256_load_si256((__m256i*)&ref_offset[0]));
			yRefUpper = _mm256_add_epi32(_mm256_madd_epi16(yRefUpper, yRefMulti), _mm256_load_si256((__m256i*)&ref_offset[8]));

			_mm256_store_si256((__m256i*)&ref_buffer[0], _mm256_multi6_epi32(yRefLower));
			_mm256_store_si256((__m256i*)&ref_buffer[8], _mm256_multi6_epi32(yRefUpper));

			gather_ycp(ycp_buffer, ycp_src, ref_buffer, 0);
			gather_ycp(ycp_buffer, ycp_src, ref_buffer, 4);
			createRandAVX2_1(dither, &gen_rand, ditherYC2, ditherYC);

			__m256i yYCPRef0, yYCPRef1, xYCPAvg, yYCPDiff, yYCP, yThreshold, yBase, yMask, yDither;

			yYCPRef0 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer +   0));
			yYCPRef1 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer +  96));
			xYCPAvg  = _mm256_srai_epi16(_mm256_adds_epi16(_mm256_adds_epi16(yYCPRef0, yYCPRef1), yOne256), 1);
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src +  0));
			yYCPDiff = (blur_first) ? _mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPAvg))
									: _mm256_max_epi16(_mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0)),
									                   _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef1)) );
			yThreshold = _mm256_load_si256((__m256i*)(threshold +  0));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, xYCPAvg, yMask);  //r = (mask0 & 0xff) ? b : a
			yDither  = _mm256_load_si256((__m256i *)(dither +  0));
			yYCP     = _mm256_adds_epi16(yBase, yDither);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst +  0), yYCP);
			
			gather_ycp(ycp_buffer, ycp_src, ref_buffer,  8);
			gather_ycp(ycp_buffer, ycp_src, ref_buffer, 12);
			createRandAVX2_2(dither, ref, &gen_rand, ditherYC2, ditherYC, yRangeYLimit, yRangeXLimit0, yRangeXLimit1);

			yYCPRef0 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer +  32));
			yYCPRef1 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer + 128));
			xYCPAvg  = _mm256_srai_epi16(_mm256_adds_epi16(_mm256_adds_epi16(yYCPRef0, yYCPRef1), yOne256), 1);
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 32));
			yYCPDiff = (blur_first) ? _mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPAvg))
									: _mm256_max_epi16(_mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0)),
									                   _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef1)) );
			yThreshold = _mm256_load_si256((__m256i*)(threshold + 16));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, xYCPAvg, yMask);  //r = (mask0 & 0xff) ? b : a
			yDither  = _mm256_load_si256((__m256i *)(dither + 16));
			yYCP     = _mm256_adds_epi16(yBase, yDither);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 32), yYCP);
			
			

			yYCPRef0 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer +  64));
			yYCPRef1 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer + 160));
			xYCPAvg  = _mm256_srai_epi16(_mm256_adds_epi16(_mm256_adds_epi16(yYCPRef0, yYCPRef1), yOne256), 1);
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 64));
			yYCPDiff = (blur_first) ? _mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPAvg))
									: _mm256_max_epi16(_mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0)),
									                   _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef1)) );
			yThreshold = _mm256_load_si256((__m256i*)(threshold + 32));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, xYCPAvg, yMask);  //r = (mask0 & 0xff) ? b : a
			yDither  = _mm256_load_si256((__m256i *)(dither + 32));
			yYCP     = _mm256_adds_epi16(yBase, yDither);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 64), yYCP);

			i_step = limit_1_to_16(x);
		}
	}
	//最後のライン
	if (y < y_end) {
		PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w;
		PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w;
		for (int i_step = 0, x = width - 16; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {

			__m256i yYCP0, yYCP1, yYCP2, yDither0, yDither1, yDither2;
			createRandAVX2_4(&gen_rand, ditherYC2, ditherYC, yDither0, yDither1, yDither2);

			yYCP0    = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src +  0));
			yYCP1    = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 32));
			yYCP2    = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 64));
			yYCP0    = _mm256_adds_epi16(yYCP0, yDither0);
			yYCP1    = _mm256_adds_epi16(yYCP1, yDither1);
			yYCP2    = _mm256_adds_epi16(yYCP2, yDither2);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst +  0), yYCP0);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 32), yYCP1);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 64), yYCP2);

			i_step = limit_1_to_16(x);
		}
	}
	_mm256_zeroupper();
	_mm_empty();
	band.gen_rand_avx2[thread_id] = gen_rand;
}
#endif
void decrease_banding_mode1_blur_first_p_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
	decrease_banding_mode1_avx2(thread_id, thread_num, fp, fpip, TRUE, FALSE, AVX2|AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode1_blur_first_i_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
	decrease_banding_mode1_avx2(thread_id, thread_num, fp, fpip, TRUE, TRUE, AVX2|AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode1_blur_later_p_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
	decrease_banding_mode1_avx2(thread_id, thread_num, fp, fpip, FALSE, FALSE, AVX2|AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode1_blur_later_i_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
	decrease_banding_mode1_avx2(thread_id, thread_num, fp, fpip, FALSE, TRUE, AVX2|AVX|SSE41|SSSE3|SSE2);
}

//blur_first、process_per_field、simdは定数として与え、
//条件分岐をコンパイル時に削除させる
static void __forceinline decrease_banding_mode2_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip, BOOL blur_first,  BOOL process_per_field, DWORD simd) {
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
	const int  y_start         = ( height *  thread_id    ) / thread_num;
	const int  y_end           = ( height * (thread_id+1) ) / thread_num;
	__m256i yRefMulti  = _mm256_unpacklo_epi16(_mm256_set1_epi16(max_w), yOne256);
	__m256i yRefMulti2 = _mm256_unpacklo_epi16(yOne256, _mm256_set1_epi16(-max_w));
	
	BYTE     __declspec(align(32)) ref[32];
	short    __declspec(align(32)) dither[48];
	short    __declspec(align(32)) ditherYC[48];
	short    __declspec(align(32)) ditherYC2[48];
	PIXEL_YC __declspec(align(32)) ycp_buffer[64];
	int      __declspec(align(32)) ref_buffer[32];
	short    __declspec(align(32)) threshold[48];

	for (int i = 0; i < 16; i++) {
		threshold[3*i+0] = threshold_y;
		threshold[3*i+1] = threshold_cb;
		threshold[3*i+2] = threshold_cr;
	}
	for (int i = 0; i < 16; i++) {
		ditherYC[3*i+0] = ditherY;
		ditherYC[3*i+1] = ditherC;
		ditherYC[3*i+2] = ditherC;
	}
	{
		__m256i y0 = _mm256_load_si256((__m256i *)(ditherYC +  0));
		__m256i y1 = _mm256_load_si256((__m256i *)(ditherYC + 16));
		__m256i y2 = _mm256_load_si256((__m256i *)(ditherYC + 32));
		y0 = _mm256_slli_epi16(y0, 1);
		y1 = _mm256_slli_epi16(y1, 1);
		y2 = _mm256_slli_epi16(y2, 1);
		y0 = _mm256_add_epi16(y0, yOne256);
		y1 = _mm256_add_epi16(y1, yOne256);
		y2 = _mm256_add_epi16(y2, yOne256);
		_mm256_store_si256((__m256i *)(ditherYC2 +  0), y0);
		_mm256_store_si256((__m256i *)(ditherYC2 + 16), y1);
		_mm256_store_si256((__m256i *)(ditherYC2 + 32), y2);
	}
	xor514x2_t gen_rand;
	if (!rand_each_frame) {
		xor514x2_init(&gen_rand, seed + (y_start << 2));
	} else {
		gen_rand = band.gen_rand_avx2[thread_id];
	}


	int y;
	const int y_main_end = min(y_end, height - 1);
	for (y = y_start; y < y_main_end; y++) {
		PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w;
		PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w;
		__m256i yRangeYLimit = _mm256_set1_epi16(min3(range, y, height - y - 1));
		__m256i yRangeXLimit0 = _mm256_load_si256((__m256i*)x_range_offset);
		__m256i yRangeXLimit1 = _mm256_subs_epu16(_mm256_set1_epi16(width - 1), _mm256_load_si256((__m256i*)x_range_offset));
		createRandAVX2_0(ref, &gen_rand, yRangeYLimit, yRangeXLimit0, yRangeXLimit1);
		for (int i_step = 0, x = width - 16; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {
			__m256i yRef = _mm256_loadu_si256((__m256i*)ref);

			if (process_per_field) {
				__m256i yRef2 = apply_field_mask_256(yRef, TRUE);
				__m256i yRefUpper = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(yRef2, 1));
				__m256i yRefLower = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(yRef2));
			
				_mm256_store_si256((__m256i*)&ref_buffer[ 0], _mm256_multi6_epi32(_mm256_add_epi32(_mm256_madd_epi16(yRefLower, yRefMulti), _mm256_load_si256((__m256i*)&ref_offset[0]))));
				_mm256_store_si256((__m256i*)&ref_buffer[ 8], _mm256_multi6_epi32(_mm256_add_epi32(_mm256_madd_epi16(yRefUpper, yRefMulti), _mm256_load_si256((__m256i*)&ref_offset[8]))));

				yRef2 = apply_field_mask_256(yRef, FALSE);
				yRefUpper = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(yRef2, 1));
				yRefLower = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(yRef2));

				_mm256_store_si256((__m256i*)&ref_buffer[16], _mm256_multi6_epi32(_mm256_add_epi32(_mm256_madd_epi16(yRefLower, yRefMulti2), _mm256_load_si256((__m256i*)&ref_offset[0]))));
				_mm256_store_si256((__m256i*)&ref_buffer[24], _mm256_multi6_epi32(_mm256_add_epi32(_mm256_madd_epi16(yRefUpper, yRefMulti2), _mm256_load_si256((__m256i*)&ref_offset[8]))));
			} else {
				__m256i yRefUpper = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(yRef, 1));
				__m256i yRefLower = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(yRef));
			
				_mm256_store_si256((__m256i*)&ref_buffer[ 0], _mm256_multi6_epi32(_mm256_add_epi32(_mm256_madd_epi16(yRefLower, yRefMulti), _mm256_load_si256((__m256i*)&ref_offset[0]))));
				_mm256_store_si256((__m256i*)&ref_buffer[ 8], _mm256_multi6_epi32(_mm256_add_epi32(_mm256_madd_epi16(yRefUpper, yRefMulti), _mm256_load_si256((__m256i*)&ref_offset[8]))));

				_mm256_store_si256((__m256i*)&ref_buffer[16], _mm256_multi6_epi32(_mm256_add_epi32(_mm256_madd_epi16(yRefLower, yRefMulti2), _mm256_load_si256((__m256i*)&ref_offset[0]))));
				_mm256_store_si256((__m256i*)&ref_buffer[24], _mm256_multi6_epi32(_mm256_add_epi32(_mm256_madd_epi16(yRefUpper, yRefMulti2), _mm256_load_si256((__m256i*)&ref_offset[8]))));
			}

			
			gather_ycp(ycp_buffer +  0, ycp_src, ref_buffer +  0, 0);
			gather_ycp(ycp_buffer +  0, ycp_src, ref_buffer +  0, 4);
			createRandAVX2_1(dither, &gen_rand, ditherYC2, ditherYC);
			gather_ycp(ycp_buffer + 32, ycp_src, ref_buffer + 16, 0);
			gather_ycp(ycp_buffer + 32, ycp_src, ref_buffer + 16, 4);

			__m256i yYCPRef0, yYCPRef1, xYCPRef2, xYCPRef3;
			__m256i xYCPAvg, yYCPDiff, yYCP, yThreshold, yBase, yMask, yDither;

			yYCPRef0 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer +   0));
			yYCPRef1 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer +  96));
			xYCPRef2 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer + 192));
			xYCPRef3 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer + 288));
			xYCPAvg  = _mm256_srai_epi16(_mm256_adds_epi16(yTwo256,
				                      _mm256_adds_epi16(_mm256_adds_epi16(yYCPRef0, yYCPRef1),
				                                        _mm256_adds_epi16(xYCPRef2, xYCPRef3))), 2);
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src +  0));
			yYCPDiff = (blur_first) ? _mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPAvg))
									: _mm256_max_epi16(_mm256_max_epi16(_mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0)),
									                                    _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef1))),
													   _mm256_max_epi16(_mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPRef2)),
									                                    _mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPRef3))));
			yThreshold = _mm256_load_si256((__m256i*)(threshold +  0));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, xYCPAvg, yMask);  //r = (mask0 & 0xff) ? b : a
			yDither  = _mm256_load_si256((__m256i *)(dither +  0));
			yYCP     = _mm256_adds_epi16(yBase, yDither);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst +  0), yYCP);
			
			gather_ycp(ycp_buffer +  0, ycp_src, ref_buffer +  0,  8);
			gather_ycp(ycp_buffer +  0, ycp_src, ref_buffer +  0, 12);
			createRandAVX2_2(dither, ref, &gen_rand, ditherYC2, ditherYC, yRangeYLimit, yRangeXLimit0, yRangeXLimit1);
			gather_ycp(ycp_buffer + 32, ycp_src, ref_buffer + 16,  8);
			gather_ycp(ycp_buffer + 32, ycp_src, ref_buffer + 16, 12);

			yYCPRef0 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer +  32));
			yYCPRef1 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer + 128));
			xYCPRef2 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer + 224));
			xYCPRef3 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer + 320));
			xYCPAvg  = _mm256_srai_epi16(_mm256_adds_epi16(yTwo256,
				                      _mm256_adds_epi16(_mm256_adds_epi16(yYCPRef0, yYCPRef1),
				                                        _mm256_adds_epi16(xYCPRef2, xYCPRef3))), 2);
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 32));
			yYCPDiff = (blur_first) ? _mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPAvg))
									: _mm256_max_epi16(_mm256_max_epi16(_mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0)),
									                                    _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef1))),
													   _mm256_max_epi16(_mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPRef2)),
									                                    _mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPRef3))));
			yThreshold = _mm256_load_si256((__m256i*)(threshold + 16));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, xYCPAvg, yMask);  //r = (mask0 & 0xff) ? b : a
			yDither  = _mm256_load_si256((__m256i *)(dither + 16));
			yYCP     = _mm256_adds_epi16(yBase, yDither);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 32), yYCP);



			yYCPRef0 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer +  64));
			yYCPRef1 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer + 160));
			xYCPRef2 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer + 256));
			xYCPRef3 = _mm256_load_si256((__m256i*)((BYTE*)ycp_buffer + 352));
			xYCPAvg  = _mm256_srai_epi16(_mm256_adds_epi16(yTwo256,
				                      _mm256_adds_epi16(_mm256_adds_epi16(yYCPRef0, yYCPRef1),
				                                        _mm256_adds_epi16(xYCPRef2, xYCPRef3))), 2);
			yYCP     = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 64));
			yYCPDiff = (blur_first) ? _mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPAvg))
									: _mm256_max_epi16(_mm256_max_epi16(_mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef0)),
									                                    _mm256_abs_epi16(_mm256_subs_epi16(yYCP, yYCPRef1))),
													   _mm256_max_epi16(_mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPRef2)),
									                                    _mm256_abs_epi16(_mm256_subs_epi16(yYCP, xYCPRef3))));
			yThreshold = _mm256_load_si256((__m256i*)(threshold + 32));
			yMask    = _mm256_cmpgt_epi16(yThreshold, yYCPDiff); //(a > b) ? 0xffff : 0x0000;
			yBase    = _mm256_blendv_epi8(yYCP, xYCPAvg, yMask);  //r = (mask0 & 0xff) ? b : a
			yDither  = _mm256_load_si256((__m256i *)(dither + 32));
			yYCP     = _mm256_adds_epi16(yBase, yDither);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 64), yYCP);

			i_step = limit_1_to_16(x);
		}
	}
	//最後のライン
	if (y < y_end) {
		PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w;
		PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w;
		for (int i_step = 0, x = width - 16; x >= 0; x -= i_step, ycp_src += i_step, ycp_dst += i_step) {
			
			__m256i yYCP0, yYCP1, yYCP2, yDither0, yDither1, yDither2;
			createRandAVX2_4(&gen_rand, ditherYC2, ditherYC, yDither0, yDither1, yDither2);

			yYCP0    = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src +  0));
			yYCP1    = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 32));
			yYCP2    = _mm256_loadu_si256((__m256i*)((BYTE *)ycp_src + 64));
			yYCP0    = _mm256_adds_epi16(yYCP0, yDither0);
			yYCP1    = _mm256_adds_epi16(yYCP1, yDither1);
			yYCP2    = _mm256_adds_epi16(yYCP2, yDither2);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst +  0), yYCP0);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 32), yYCP1);
			_mm256_storeu_si256((__m256i*)((BYTE *)ycp_dst + 64), yYCP2);

			i_step = limit_1_to_16(x);
		}
	}
	_mm256_zeroupper();
	_mm_empty();
	band.gen_rand_avx2[thread_id] = gen_rand;
}

void decrease_banding_mode2_blur_first_p_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
	decrease_banding_mode2_avx2(thread_id, thread_num, fp, fpip, TRUE, FALSE, AVX2|AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode2_blur_first_i_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
	decrease_banding_mode2_avx2(thread_id, thread_num, fp, fpip, TRUE, TRUE, AVX2|AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode2_blur_later_p_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
	decrease_banding_mode2_avx2(thread_id, thread_num, fp, fpip, FALSE, FALSE, AVX2|AVX|SSE41|SSSE3|SSE2);
}

void decrease_banding_mode2_blur_later_i_avx2(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
	decrease_banding_mode2_avx2(thread_id, thread_num, fp, fpip, FALSE, TRUE, AVX2|AVX|SSE41|SSSE3|SSE2);
}
