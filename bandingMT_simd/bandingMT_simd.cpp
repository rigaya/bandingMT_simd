//  -----------------------------------------------------------------------------------------
//    バンディング低減MT SIMD by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------


#include <windows.h>
#include <process.h>
#include <algorithm>

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
    return TRUE;
}

BOOL func_exit( FILTER *fp ) {
    if (band.gen_rand_avx2)
        _aligned_free(band.gen_rand_avx2);
    band.gen_rand_avx2 = NULL;
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
    if (band.availableSIMD & AVX2) {
        for (int i = 0; i < band.thread_num; i++)
            xor514x2_init(&band.gen_rand_avx2[i], band._seed + (i << 2));
    } else {
        for (int i = 0; i < band.thread_num; i++)
            xor514_init(&band.gen_rand[i], band._seed + (i << 2));
    }
}

BOOL init_band(FILTER *fp, FILTER_PROC_INFO *fpip) {
    if (NULL == band.gen_rand_avx2) {
        band.availableSIMD = get_availableSIMD() & ~AVX2;
        //使用する関数を設定
        band_set_func();
        //スレッド数の取得
        fp->exfunc->exec_multi_thread_func(multi_thread_get_thread_num, fp, fpip);
        //乱数用メモリ確保
        band.thread_num = band.current_thread_num;
        if (NULL == (band.gen_rand_avx2 = (xor514x2_t *)_aligned_malloc(sizeof(xor514x2_t) * band.thread_num, 64)))
            return FALSE;
        init_gen_rand();
    }
    return TRUE;
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
        band.decrease_banding[sample_mode].set[process_per_field][blur_first](thread_id, min(thread_num, band.current_thread_num), fp, fpip);
}
//---------------------------------------------------------------------
//        フィルタ処理関数
//---------------------------------------------------------------------
BOOL func_proc( FILTER *fp, FILTER_PROC_INFO *fpip )
{
    //初期化確認
    if (NULL == band.gen_rand_avx2) {
        init_band(fp, fpip);
    }

    const int seed = fp->track[7];
    if (seed != band._seed) {
        //設定が変わっていたら、設定を変更した後再度乱数生成
        band._seed = seed;
        init_gen_rand();
    }
    
    //    マルチスレッドでフィルタ処理関数を呼ぶ
    fp->exfunc->exec_multi_thread_func(multi_thread_func, fp, fpip);

    //    もし画像領域ポインタの入れ替えや解像度変更等の
    //    fpip の内容を変える場合はこちらの関数内で処理をする
    std::swap(fpip->ycp_edit, fpip->ycp_temp);

    //スレッド数を確認
    //増えていれば対応し、次回からそのスレッド数で対応できるようにする
    if (band.thread_num < band.current_thread_num) {
        if (band.gen_rand_avx2)
            _aligned_free(band.gen_rand_avx2);
        //乱数用メモリ確保
        band.thread_num = band.current_thread_num;
        if (NULL == (band.gen_rand_avx2 = (xor514x2_t *)_aligned_malloc(sizeof(xor514x2_t) * band.thread_num, 64)))
            return FALSE;
        init_gen_rand();
    }

    return TRUE;
}
