//  -----------------------------------------------------------------------------------------
//    バンディング低減MT SIMD by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
#include <intrin.h>
#include "banding.h"

DWORD get_availableSIMD() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    DWORD simd = NONE;
    if (CPUInfo[3] & 0x04000000) simd |= SSE2;
    if (CPUInfo[2] & 0x00000001) simd |= SSE3;
    if (CPUInfo[2] & 0x00000200) simd |= SSSE3;
    if (CPUInfo[2] & 0x00080000) simd |= SSE41;
    if (CPUInfo[2] & 0x00100000) simd |= SSE42;
    if (CPUInfo[2] & 0x00800000) simd |= POPCNT;
#if (_MSC_VER >= 1600)
    UINT64 xgetbv = 0;
    if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
        xgetbv = _xgetbv(0);
        if ((xgetbv & 0x06) == 0x06)
            simd |= AVX;
    }
#endif
#if (_MSC_VER >= 1700)
    __cpuid(CPUInfo, 7);
    if ((simd & AVX) && (CPUInfo[1] & 0x00000020))
        simd |= AVX2;
#endif
    return simd;
}

DWORD cpu_core_count() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
}

#include "banding.h"

static const func_decrease_banding_mode_t mode0_c = {
    decrease_banding_mode0_c, decrease_banding_mode0_c,
    decrease_banding_mode0_c, decrease_banding_mode0_c
};
static const func_decrease_banding_mode_t mode0_sse2 = {
    decrease_banding_mode0_p_sse2, decrease_banding_mode0_p_sse2,
    decrease_banding_mode0_i_sse2, decrease_banding_mode0_i_sse2
};
static const func_decrease_banding_mode_t mode0_ssse3 = {
    decrease_banding_mode0_p_ssse3, decrease_banding_mode0_p_ssse3,
    decrease_banding_mode0_i_ssse3, decrease_banding_mode0_i_ssse3,
};
static const func_decrease_banding_mode_t mode0_sse41 = {
    decrease_banding_mode0_p_sse41, decrease_banding_mode0_p_sse41,
    decrease_banding_mode0_i_sse41, decrease_banding_mode0_i_sse41,
};
static const func_decrease_banding_mode_t mode0_avx = {
    decrease_banding_mode0_p_avx, decrease_banding_mode0_p_avx,
    decrease_banding_mode0_i_avx, decrease_banding_mode0_i_avx,
};
static const func_decrease_banding_mode_t mode0_avx2 = {
    decrease_banding_mode0_p_avx2, decrease_banding_mode0_p_avx2,
    decrease_banding_mode0_i_avx2, decrease_banding_mode0_i_avx2,
};
    
static const func_decrease_banding_mode_t mode12_c = {
    decrease_banding_mode12_c, decrease_banding_mode12_c,
    decrease_banding_mode12_c, decrease_banding_mode12_c,
};
static const func_decrease_banding_mode_t mode1_sse2 = {
    decrease_banding_mode1_blur_later_p_sse2, decrease_banding_mode1_blur_first_p_sse2,
    decrease_banding_mode1_blur_later_i_sse2, decrease_banding_mode1_blur_first_i_sse2,
};
static const func_decrease_banding_mode_t mode1_ssse3 = {
    decrease_banding_mode1_blur_later_p_ssse3, decrease_banding_mode1_blur_first_p_ssse3,
    decrease_banding_mode1_blur_later_i_ssse3, decrease_banding_mode1_blur_first_i_ssse3,
};
static const func_decrease_banding_mode_t mode1_sse41 = {
    decrease_banding_mode1_blur_later_p_sse41, decrease_banding_mode1_blur_first_p_sse41,
    decrease_banding_mode1_blur_later_i_sse41, decrease_banding_mode1_blur_first_i_sse41,
};
static const func_decrease_banding_mode_t mode1_avx = {
    decrease_banding_mode1_blur_later_p_avx, decrease_banding_mode1_blur_first_p_avx,
    decrease_banding_mode1_blur_later_i_avx, decrease_banding_mode1_blur_first_i_avx,
};
static const func_decrease_banding_mode_t mode1_avx2 = {
    decrease_banding_mode1_blur_later_p_avx2, decrease_banding_mode1_blur_first_p_avx2,
    decrease_banding_mode1_blur_later_i_avx2, decrease_banding_mode1_blur_first_i_avx2,
};
static const func_decrease_banding_mode_t mode2_sse2 = {
    decrease_banding_mode2_blur_later_p_sse2, decrease_banding_mode2_blur_first_p_sse2,
    decrease_banding_mode2_blur_later_i_sse2, decrease_banding_mode2_blur_first_i_sse2,
};
static const func_decrease_banding_mode_t mode2_ssse3 = {
    decrease_banding_mode2_blur_later_p_ssse3, decrease_banding_mode2_blur_first_p_ssse3,
    decrease_banding_mode2_blur_later_i_ssse3, decrease_banding_mode2_blur_first_i_ssse3,
};
static const func_decrease_banding_mode_t mode2_sse41 = {
    decrease_banding_mode2_blur_later_p_sse41, decrease_banding_mode2_blur_first_p_sse41,
    decrease_banding_mode2_blur_later_i_sse41, decrease_banding_mode2_blur_first_i_sse41,
};
static const func_decrease_banding_mode_t mode2_avx = {
    decrease_banding_mode2_blur_later_p_avx, decrease_banding_mode2_blur_first_p_avx,
    decrease_banding_mode2_blur_later_i_avx, decrease_banding_mode2_blur_first_i_avx,
};
static const func_decrease_banding_mode_t mode2_avx2 = {
    decrease_banding_mode2_blur_later_p_avx2, decrease_banding_mode2_blur_first_p_avx2,
    decrease_banding_mode2_blur_later_i_avx2, decrease_banding_mode2_blur_first_i_avx2,
};


void band_set_func() {
    const DWORD simd_avail = get_availableSIMD();
    const func_decrease_banding_mode_t deband[][6] = {
        { mode0_c,  mode0_sse2, mode0_ssse3, mode0_sse41, mode0_avx, mode0_avx2 },
        { mode12_c, mode1_sse2, mode1_ssse3, mode1_sse41, mode1_avx, mode1_avx2 },
        { mode12_c, mode2_sse2, mode2_ssse3, mode2_sse41, mode2_avx, mode2_avx2 },
    };
    const int simd_idx = !!(simd_avail & SSE2) + !!(simd_avail & SSSE3) + !!(simd_avail & SSE41) + !!(simd_avail & AVX) + !!(simd_avail & AVX2);
    band.decrease_banding[0] = deband[0][simd_idx];
    band.decrease_banding[1] = deband[1][simd_idx];
    band.decrease_banding[2] = deband[2][simd_idx];
}


#include "xor_rand.h"

// ランダムな128bit列をランダムな -range ～ range にして返す
// range は0～127以下
static inline char random_range(BYTE random, char range) {
    return ((((range << 1) + 1) * (int)random) >> 8) - range;
}

#include "filter.h"

static inline PIXEL_YC get_diff_abs(PIXEL_YC a, PIXEL_YC b) {
    PIXEL_YC diff;
    diff.y  = abs(a.y  - b.y);
    diff.cb = abs(a.cb - b.cb);
    diff.cr = abs(a.cr - b.cr);
    return diff;
}

static inline PIXEL_YC get_avg(PIXEL_YC a, PIXEL_YC b) {
    PIXEL_YC avg;
    avg.y  = (a.y  + b.y  + 1) >> 1;
    avg.cb = (a.cb + b.cb + 1) >> 1;
    avg.cr = (a.cr + b.cr + 1) >> 1;
    return avg;
}

static inline PIXEL_YC get_avg(PIXEL_YC a, PIXEL_YC b, PIXEL_YC c, PIXEL_YC d) {
    PIXEL_YC avg;
    avg.y  = (a.y  + b.y  + c.y  + d.y  + 2) >> 2;
    avg.cb = (a.cb + b.cb + c.cb + d.cb + 2) >> 2;
    avg.cr = (a.cr + b.cr + c.cr + d.cr + 2) >> 2;
    return avg;
}

static inline PIXEL_YC get_max(PIXEL_YC a, PIXEL_YC b) {
    PIXEL_YC max_value;
    max_value.y  = max(a.y,  b.y );
    max_value.cb = max(a.cb, b.cb);
    max_value.cr = max(a.cr, b.cr);
    return max_value;
}

static inline int min4(int a, int b, int c, int d) {
    return min(min(a, b), min(c, d));
}

static inline PIXEL_YC get_max(PIXEL_YC a, PIXEL_YC b, PIXEL_YC c, PIXEL_YC d) {
    return get_max(get_max(a, b), get_max(c, d));
}

void decrease_banding_mode0_c(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    const int max_w  = fpip->max_w;
    const int width = fpip->w;
    const int height = fpip->h;
    const int seed    = fp->track[7];
    const int  rand_each_frame = fp->check[1];
    const int  sample_mode     = fp->track[6];
    const int  blur_first      = fp->check[0];
    const BYTE range           = fp->track[0];
    const int  threshold_y     = fp->track[1] << (!(sample_mode && blur_first) + 1);
    const int  threshold_cb    = fp->track[2] << (!(sample_mode && blur_first) + 1);
    const int  threshold_cr    = fp->track[3] << (!(sample_mode && blur_first) + 1);
    const int  y_start         = ( height *  thread_id    ) / thread_num;
    const int  y_end           = ( height * (thread_id+1) ) / thread_num;
    xor128_t gen_rand;
    xor128_init(&gen_rand, seed + (y_start << 2) + ((fpip->frame * rand_each_frame) << 4));

    const BYTE field_mask      = fp->check[2] ? 0xfe : 0xff;
    for (int y = y_start; y < y_end; y++) {
        PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w;
        PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w;
        const int y_limit = min(y, height-y-1);
        for (int x = 0; x < width; x++, ycp_src++, ycp_dst++) {
            const BYTE range_limited = min4(range, y_limit, x, width-x-1);
            xor128(&gen_rand);
            const int refA = random_range((gen_rand.w & 0x00ff),      range_limited);
            const int refB = random_range((gen_rand.w & 0xff00) >> 8, range_limited);
            const int ref = (char)(refA & field_mask) * max_w + refB;
            PIXEL_YC diff = get_diff_abs(ycp_src[0], ycp_src[ref]);
            ycp_dst->y  = (diff.y  < threshold_y)  ? ycp_src[ref].y  : ycp_src->y;
            ycp_dst->cb = (diff.cb < threshold_cb) ? ycp_src[ref].cb : ycp_src->cb;
            ycp_dst->cr = (diff.cr < threshold_cr) ? ycp_src[ref].cr : ycp_src->cr;
        }
    }
}

void decrease_banding_mode12_c(int thread_id, int thread_num, FILTER* fp, FILTER_PROC_INFO *fpip) {
    const int max_w  = fpip->max_w;
    const int width = fpip->w;
    const int height = fpip->h;
    const int seed    = fp->track[7];
    const int ditherY = fp->track[4];
    const int ditherC = fp->track[5];
    const int  rand_each_frame = fp->check[1];
    const int  sample_mode     = fp->track[6];
    const int  blur_first      = fp->check[0];
    const BYTE range           = fp->track[0];
    const int  threshold_y     = fp->track[1] << (!(sample_mode && blur_first) + 1);
    const int  threshold_cb    = fp->track[2] << (!(sample_mode && blur_first) + 1);
    const int  threshold_cr    = fp->track[3] << (!(sample_mode && blur_first) + 1);
    const int  y_start         = ( height *  thread_id    ) / thread_num;
    const int  y_end           = ( height * (thread_id+1) ) / thread_num;
    const BYTE field_mask      = fp->check[2] ? 0xfe : 0xff;
    xor128_t gen_rand;
    xor128_init(&gen_rand, seed + (y_start << 2) + ((fpip->frame * rand_each_frame) << 4));

    for (int y = y_start; y < y_end; y++) {
        PIXEL_YC *ycp_src = fpip->ycp_edit + y * max_w;
        PIXEL_YC *ycp_dst = fpip->ycp_temp + y * max_w;
        int y_limit = min(y, height-y-1);
        for (int x = 0; x < width; x++, ycp_src++, ycp_dst++) {
            const BYTE range_limited = min4(range, y_limit, x, width-x-1);
            xor128(&gen_rand);
            const char refA = random_range((gen_rand.w & 0x00ff),      range_limited);
            const char refB = random_range((gen_rand.w & 0xff00) >> 8, range_limited);
            PIXEL_YC avg, diff;
            if (sample_mode == 1) {
                const int ref = (char)(refA & field_mask) * max_w + refB;
                avg = get_avg(ycp_src[ref], ycp_src[-ref]);
                diff = (blur_first) ? get_diff_abs(ycp_src[0], avg)
                                    : get_max(get_diff_abs(ycp_src[0], ycp_src[ ref]),
                                              get_diff_abs(ycp_src[0], ycp_src[-ref]));
            } else {
                const int ref_0 = (char)(refA & field_mask) * max_w + refB;
                const int ref_1 = refA - (char)(refB & field_mask) * max_w;
                avg = get_avg(ycp_src[ ref_0], ycp_src[-ref_0], ycp_src[ ref_1], ycp_src[-ref_1]);
                diff = (blur_first) ? get_diff_abs(ycp_src[0], avg)
                                    : get_max( get_diff_abs(ycp_src[0], ycp_src[ ref_0]),
                                               get_diff_abs(ycp_src[0], ycp_src[-ref_0]),
                                               get_diff_abs(ycp_src[0], ycp_src[ ref_1]),
                                               get_diff_abs(ycp_src[0], ycp_src[-ref_1]) );
            }
            PIXEL_YC base;
            base.y  = (diff.y  < threshold_y)  ? avg.y  : ycp_src->y;
            base.cb = (diff.cb < threshold_cb) ? avg.cb : ycp_src->cb;
            base.cr = (diff.cr < threshold_cr) ? avg.cr : ycp_src->cr;
            xor128(&gen_rand);
            ycp_dst->y  = base.y  + random_range((gen_rand.w & 0x0000ff),      ditherY);
            ycp_dst->cb = base.cb + random_range((gen_rand.w & 0x00ff00) >> 8, ditherC);
            ycp_dst->cr = base.cr + random_range((gen_rand.w & 0xff0000) >>16, ditherC);
        }
    }
}
