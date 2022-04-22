
# バンディング低減MT SIMD
  by rigaya  
  original idea by [がらくたハウスのがらくた置き場様](http://www.geocities.jp/flash3kyuu/)

バンディング低減MT SIMDは[がらくたハウスのがらくた置き場様](http://www.geocities.jp/flash3kyuu/)の
バンディング低減フィルタMTを勝手に高速化したものです。

完全に手探りででっち上げたので、怪しいところもありますが、だいたい同じ効果が得られるかと。  
AVX512-VNNI / AVX512 / AVX2+AVX-VNNI / AVX2 / 128bit-AVX / SSE4.1 / SSSE3 / SSE2 / MMX で高速化されています。
環境に合わせて、最速のものが自動的に選択されます。

## ダウンロード
[github releases](https://github.com/rigaya/bandingMT_simd/releases)

## 更新履歴
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-15.html)

## 基本動作環境
Windows 8.1, 10, 11 (x86/x64)  
Aviutl 1.00 以降

## バンディング低減MT SIMD 使用にあたっての注意事項
無保証です。自己責任で使用してください。  
バンディング低減MT SIMDを使用したことによる、いかなる損害・トラブルについても責任を負いません。  


## パラメータ
初期設定は比較的強めです。ほぼオリジナルの説明のままですが、一部変更しています。
- range  
ぼかす範囲です。この範囲内の近傍画素からサンプルを取り、ブラー処理を行います。

- Y　Cb　Cr  
各成分の閾値です。この値が高いと階調飛びを減らす一方で、細かい線などが潰れます。

- ditherY　ditherC  
sampleの設定値が1～2のときのみ反映されます。
ブラー処理後に全体的に付加するディザの強度です。
Aviutl上でのプレビューは最終出力するフォーマットよりも高精度なため、
プレビュー時に階調飛びが目立たなくても、最終出力時に階調飛びが発生することがあります。
最後までなるべくディザが残るように、圧縮効率とのバランスを取りながら設定する必要があります。

- sample
	- 設定値：0  
		周辺1画素を参照し、元の画素値を維持したまま処理を行います（ブラー処理を行わない）。
		中間階調を作らず、元の画素値がそのままディザ成分になるので、
		後段のフィルタで潰さない限り原理上バンディングは発生しませんが、
		ノイズ成分が増えやすいので、ソースの階調が比較的良好なときに使用します。
		処理はもっとも高速です。

	- 設定値：1  
		周辺1画素とその点対称画素の計2画素を参照し、ブラー処理を行います。
		階調を滑らかにしすぎた場合は、
		ditherを強くして後段でのバンディングの発生を防いでください。

	- 設定値：2  
		周辺2画素とその点対称画素の計4画素を参照し、ブラー処理を行います。
		階調を滑らかにしすぎた場合は、
		ditherを強くして後段でのバンディングの発生を防いでください。
		設定値：1と処理が異なるので閾値の変更が必要になります。

- seed  
乱数生成時のシード値を変えます。
通常必要ありません。

- ブラー処理を先に  
sampleの設定値が、1～2のときのみ反映されます。
ブラー処理を先にすることでディザ強度を減らしつつ、階調飛びが多い素材での効果を上げます。
全体的に副作用が強くなり細かい線が潰れやすくなります。
ONとOFFでは処理が異なるので閾値の変更が必要になります。

- 毎フレーム乱数を生成  
毎フレームシード値を変更します。
乱数を固定すると、うっすらと霞がかかったように見えるので、
それを避けたい場合はこの設定を利用してください。

- フィールド処理  
フィールド別で処理します。

## 謝辞
素晴らしいプラグインを公開してくださっている、[がらくたハウスのがらくた置き場様](http://www.geocities.jp/flash3kyuu/)に深く感謝いたします。

## ソースコードについて
MITライセンスです。

### ソースの構成
VCビルド  
文字コード: UTF-8-BOM  
改行: CRLF  
インデント: 空白x4  
