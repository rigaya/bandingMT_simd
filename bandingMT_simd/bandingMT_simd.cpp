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


#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#include <algorithm>
#include <map>

#define DEFINE_GLOBAL
#include "filter.h"
#include "banding.h"
#include "banding_version.h"
#include "xor_rand.h"

//---------------------------------------------------------------------
//        フィルタ構造体定義
//---------------------------------------------------------------------
#define    TRACK_N    8                                                                                    //  トラックバーの数
TCHAR    *track_name[] =        {"range",  "Y", "Cb", "Cr","ditherY", "ditherC", "sample", "seed" };    //  トラックバーの名前
int        track_default[] =    {     15,  15,   15,    15,     15,        15,         1,     0   };    //  トラックバーの初期値
int        track_s[] =            {      0,   0,    0,     0,      0,         0,         0,     0   };    //  トラックバーの下限値
int        track_e[] =            {     63,  31,   31,    31,     31,        31,         2,   127   };    //  トラックバーの上限値

#define    CHECK_N    3                                                                   //  チェックボックスの数
TCHAR    *check_name[] =     { "ブラー処理を先に","毎フレーム乱数を生成","フィールド処理" }; //  チェックボックスの名前
int         check_default[] =     { 0, 0, 0 };    //  チェックボックスの初期値 (値は0か1)

FILTER_DLL filter = {
    FILTER_FLAG_EX_INFORMATION,    //    フィルタのフラグ
                                //    FILTER_FLAG_ALWAYS_ACTIVE        : フィルタを常にアクティブにします
                                //    FILTER_FLAG_CONFIG_POPUP        : 設定をポップアップメニューにします
                                //    FILTER_FLAG_CONFIG_CHECK        : 設定をチェックボックスメニューにします
                                //    FILTER_FLAG_CONFIG_RADIO        : 設定をラジオボタンメニューにします
                                //    FILTER_FLAG_EX_DATA                : 拡張データを保存出来るようにします。
                                //    FILTER_FLAG_PRIORITY_HIGHEST    : フィルタのプライオリティを常に最上位にします
                                //    FILTER_FLAG_PRIORITY_LOWEST        : フィルタのプライオリティを常に最下位にします
                                //    FILTER_FLAG_WINDOW_THICKFRAME    : サイズ変更可能なウィンドウを作ります
                                //    FILTER_FLAG_WINDOW_SIZE            : 設定ウィンドウのサイズを指定出来るようにします
                                //    FILTER_FLAG_DISP_FILTER            : 表示フィルタにします
                                //    FILTER_FLAG_EX_INFORMATION        : フィルタの拡張情報を設定できるようにします
                                //    FILTER_FLAG_NO_CONFIG            : 設定ウィンドウを表示しないようにします
                                //    FILTER_FLAG_AUDIO_FILTER        : オーディオフィルタにします
                                //    FILTER_FLAG_RADIO_BUTTON        : チェックボックスをラジオボタンにします
                                //    FILTER_FLAG_WINDOW_HSCROLL        : 水平スクロールバーを持つウィンドウを作ります
                                //    FILTER_FLAG_WINDOW_VSCROLL        : 垂直スクロールバーを持つウィンドウを作ります
                                //    FILTER_FLAG_IMPORT                : インポートメニューを作ります
                                //    FILTER_FLAG_EXPORT                : エクスポートメニューを作ります
    0,0,                        //    設定ウインドウのサイズ (FILTER_FLAG_WINDOW_SIZEが立っている時に有効)
    AUF_FULL_NAME,    //    フィルタの名前
    TRACK_N,                    //    トラックバーの数 (0なら名前初期値等もNULLでよい)
    track_name,                    //    トラックバーの名前郡へのポインタ
    track_default,                //    トラックバーの初期値郡へのポインタ
    track_s,track_e,            //    トラックバーの数値の下限上限 (NULLなら全て0～256)
    CHECK_N,                    //    チェックボックスの数 (0なら名前初期値等もNULLでよい)
    check_name,                    //    チェックボックスの名前郡へのポインタ
    check_default,                //    チェックボックスの初期値郡へのポインタ
    func_proc,                    //    フィルタ処理関数へのポインタ (NULLなら呼ばれません)
    func_init,                    //    開始時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    func_exit,                    //    終了時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                        //    設定が変更されたときに呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                        //    設定ウィンドウにウィンドウメッセージが来た時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,NULL,                    //    システムで使いますので使用しないでください
    NULL,                        //  拡張データ領域へのポインタ (FILTER_FLAG_EX_DATAが立っている時に有効)
    NULL,                        //  拡張データサイズ (FILTER_FLAG_EX_DATAが立っている時に有効)
    AUF_VERSION_NAME,
                                //  フィルタ情報へのポインタ (FILTER_FLAG_EX_INFORMATIONが立っている時に有効)
    NULL,                        //    セーブが開始される直前に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                        //    セーブが終了した直前に呼ばれる関数へのポインタ (NULLなら呼ばれません)
};


//---------------------------------------------------------------------
//        フィルタ構造体のポインタを渡す関数
//---------------------------------------------------------------------
EXTERN_C FILTER_DLL __declspec(dllexport) * __stdcall GetFilterTable( void )
{
    return &filter;
}

BOOL func_init( FILTER *fp ) {
    ZeroMemory(&band, sizeof(band));
    band.block_count_x = 1;
    band.block_count_y = 1;
    return TRUE;
}

BOOL func_exit( FILTER *fp ) {
    if (band.gen_rand_avx512)
        _aligned_free(band.gen_rand_avx512);
    band.gen_rand_avx512 = NULL;
    return TRUE;
}

//---------------------------------------------------------------------
//        フィルタ処理関数
//-------------------------------------------------------------------
void multi_thread_get_thread_num( int thread_id, int thread_num, void *param1, void *param2 )
{
    //スレッド数を取得、保存する
    if (thread_id == 0)
        band.current_thread_num = thread_num;
}

static void init_gen_rand() {
    if (band.availableSIMD & AVX512BW) {
        for (int i = 0; i < band.thread_num; i++)
            xor514x4_init(&band.gen_rand_avx512[i], band._seed + (i << 2));
    } else if (band.availableSIMD & AVX2) {
        for (int i = 0; i < band.thread_num; i++)
            xor514x2_init(&band.gen_rand_avx2[i], band._seed + (i << 2));
    } else {
        for (int i = 0; i < band.thread_num; i++)
            xor514_init(&band.gen_rand[i], band._seed + (i << 2));
    }
}

BOOL init_band(FILTER *fp, FILTER_PROC_INFO *fpip) {
    if (NULL == band.gen_rand_avx512) {
        band.availableSIMD = get_availableSIMD();
        //使用する関数を設定
        band_set_func();
        //スレッド数の取得
        fp->exfunc->exec_multi_thread_func(multi_thread_get_thread_num, fp, fpip);
        //乱数用メモリ確保
        band.thread_num = band.current_thread_num;
        if (NULL == (band.gen_rand_avx512 = (xor514x4_t *)_aligned_malloc(sizeof(xor514x4_t) * band.thread_num, 64)))
            return FALSE;
        init_gen_rand();
    }
    return TRUE;
}

void band_get_block_range(int ib, int width, int height, int *x_start, int *x_fin, int *y_start, int *y_fin) {
    //ブロックのy方向のインデックス
    int by = ib / band.block_count_x;
    //ブロックのx方向のインデックス
    int bx = ib - by * band.block_count_x;
    //ブロックのx方向の範囲を算出
    *x_start = ((width * bx) / band.block_count_x + 31) & (~31);
    int x_fin_raw = ((width * (bx+1)) / band.block_count_x + 31) & (~31);
    *x_fin   = (bx == band.block_count_x-1) ? width : x_fin_raw;
    //ブロックのy方向の範囲を算出
    *y_start = (height * by) / band.block_count_y;
    *y_fin   = (height * (by+1)) / band.block_count_y;
}

void multi_thread_func( int thread_id, int thread_num, void *param1, void *param2 )
{
//    thread_id    : スレッド番号 ( 0 ～ thread_num-1 )
//    thread_num    : スレッド数 ( 1 ～ )
//    param1        : 汎用パラメータ
//    param2        : 汎用パラメータ
//
//    この関数内からWin32APIや外部関数(rgb2yc,yc2rgbは除く)を使用しないでください。
//
    FILTER *fp                = (FILTER *)param1;
    FILTER_PROC_INFO *fpip    = (FILTER_PROC_INFO *)param2;
    
    //スレッド数を取得、保存する
    if (thread_id == 0)
        band.current_thread_num = thread_num;

    //設定
    const int sample_mode       =   fp->track[6];
    const int blur_first        = !!fp->check[0];
    const int process_per_field = !!fp->check[2];

    //バンディング低減 (確保しているスレッド数分だけ実行する)
    //使用スレッド数の拡大は、このmulti_thread_funcが終了後に対応
    if (thread_id < band.current_thread_num)
        band.decrease_banding[sample_mode].set[process_per_field][blur_first](thread_id, std::min(thread_num, band.current_thread_num), fp, fpip);
}

#if BANDING_PERF_CHECK
static char **dummy_area = nullptr;
static int dummy_area_count = 0;
static const int DUMMY_AREA = 4 * 1024 * 1024;

void multi_thread_func_dummy(int thread_id, int thread_num, void *param1, void *param2) {
    memcpy(dummy_area[thread_id] + DUMMY_AREA, dummy_area[thread_id], DUMMY_AREA);
}

void band_perf_check(FILTER *fp, FILTER_PROC_INFO *fpip) {
    static int check_count = 0;
    static std::map<int64_t, double> check_ms;
    static const int X_DIV[] = { 1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40 };
    static const int Y_DIV[] = { 1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 96, 128 };
    const int sample_mode    = fp->track[6];
    int64_t nFreq, nStart, nFin;
    if (dummy_area_count < band.current_thread_num) {
        for (int i = 0; i < dummy_area_count; i++) {
            if (dummy_area[i]) {
                _aligned_free(dummy_area[i]);
            }
        }
        if (dummy_area) {
            free(dummy_area);
            dummy_area = nullptr;
        }
        dummy_area_count = band.current_thread_num;
        dummy_area = (char **)malloc(sizeof(char *) * band.current_thread_num);
        for (int i = 0; i < band.current_thread_num; i++) {
            dummy_area[i] = (char *)_aligned_malloc(2 * DUMMY_AREA, 32);
        }
    }
    auto gen_key = [](int sample_mode, int block_count_y, int block_count_x) {
        return ((int64_t)sample_mode) << 32 | ((int64_t)block_count_y << 16) | (int64_t)block_count_x;
    };
    QueryPerformanceFrequency((LARGE_INTEGER *)&nFreq);
    for (int y = 0; y < _countof(Y_DIV); y++) {
        for (int x = 0; x < _countof(X_DIV); x++) {
            band.block_count_x = X_DIV[x];
            band.block_count_y = Y_DIV[y];
            //スレッド数以上の並列度があれば計測
            if (band.block_count_x * band.block_count_y >= band.current_thread_num) {
                //一度、フレームデータをキャッシュから追い出してから計測
                fp->exfunc->exec_multi_thread_func(multi_thread_func_dummy, nullptr, nullptr);
                //計測開始
                QueryPerformanceCounter((LARGE_INTEGER *)&nStart);
                fp->exfunc->exec_multi_thread_func(multi_thread_func, fp, fpip);
                QueryPerformanceCounter((LARGE_INTEGER *)&nFin);
                int64_t key = gen_key(sample_mode, band.block_count_y, band.block_count_x);
                check_ms[key] += (nFin - nStart) * 1000.0 / (double)nFreq;
            }
        }
    }
    check_count++;
    if (check_count % 256 == 0) {
        FILE *fpout = NULL;
        if (0 == fopen_s(&fpout, "bandingSIMD_perf_check.csv", "w")) {
            fprintf(fpout, "w,%d,h,%d,count,%d\n", fpip->w, fpip->h, check_count);
            for (int x = 0; x < _countof(X_DIV); x++) {
                fprintf(fpout, ",%d", X_DIV[x]);
            }
            fprintf(fpout, "\n");
            for (int y = 0; y < _countof(Y_DIV); y++) {
                fprintf(fpout, "%d,", Y_DIV[y]);
                for (int x = 0; x < _countof(X_DIV); x++) {
                    int64_t key = gen_key(sample_mode, Y_DIV[y], X_DIV[x]);
                    //値があれば出力
                    if (check_ms.count(key)) {
                        fprintf(fpout, "%.4f,", check_ms[key] / check_count);
                    } else {
                        fprintf(fpout, ",");
                    }
                }
                fprintf(fpout, "\n");
            }
            fclose(fpout);
        }
    }
}
#endif

//---------------------------------------------------------------------
//        フィルタ処理関数
//---------------------------------------------------------------------
BOOL func_proc( FILTER *fp, FILTER_PROC_INFO *fpip )
{
    //初期化確認
    if (NULL == band.gen_rand_avx512) {
        init_band(fp, fpip);
    }

    const int seed = fp->track[7];
    if (seed != band._seed) {
        //設定が変わっていたら、設定を変更した後再度乱数生成
        band._seed = seed;
        init_gen_rand();
    }
    band.block_count_x = (fpip->w + 127) / 128;
    band.block_count_y = band.current_thread_num;
    
    //    マルチスレッドでフィルタ処理関数を呼ぶ
    fp->exfunc->exec_multi_thread_func(multi_thread_func, fp, fpip);

#if BANDING_PERF_CHECK
    band_perf_check(fp, fpip);
#endif

    //    もし画像領域ポインタの入れ替えや解像度変更等の
    //    fpip の内容を変える場合はこちらの関数内で処理をする
    std::swap(fpip->ycp_edit, fpip->ycp_temp);

    //スレッド数を確認
    //増えていれば対応し、次回からそのスレッド数で対応できるようにする
    if (band.thread_num < band.current_thread_num) {
        if (band.gen_rand_avx512)
            _aligned_free(band.gen_rand_avx512);
        //乱数用メモリ確保
        band.thread_num = band.current_thread_num;
        if (NULL == (band.gen_rand_avx512 = (xor514x4_t *)_aligned_malloc(sizeof(xor514x4_t) * band.thread_num, 64)))
            return FALSE;
        init_gen_rand();
    }

    return TRUE;
}
