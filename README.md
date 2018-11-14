animeCNN
===

chainerを用いて、日本のアニメーションのキャラクターを識別するプログラム
付随してTensorFlowで同様のことを可能にしました。

お好きにコメントをください。
また、プログラムのコーディングについてご意見も頂けたらと思います。

## Description

chainerの畳み込みニューラルネットワークを使用してキャラクターを識別するためのデモプログラムです。
動画からキャラクターの画像を出力し、それらを個別に分類することで各キャラクターを下記のように出力することが可能となります。

また、現在リファクタリングを行なっています。ご了承ください。
※現在、TensorFlowはargmentに対応していません！！！
※近日対応させるので少々お待ちください

## Demo

インプットする画像
![input](https://user-images.githubusercontent.com/16191865/27992437-5c62ed7c-64cf-11e7-9d78-1cdf394b5142.jpg)

アウトプットされる画像
![output](https://user-images.githubusercontent.com/16191865/27992438-5c687328-64cf-11e7-9baf-04de561201f2.jpg)

## Requirement
 Python 3.6

## Usage

1.動画から画像を出力

2.画像を仕分ける
適当なディレクトリを作成して、そこに各キャラの名前とともにキャラクターの画像を入れていく。

3.画像をカサ増しする

python3 imgSmaill

4.学習させる

python3 bake_train.py

5.実行

python3 bake_sample.py

## Install

最低限
- openCV3.2

GPU使用時
- cuda
- or
- tensorflow-gpu

下記のアプリケーションをpipすること
- six
- tqdm
- image

- chainer
- or
- tensorflow
