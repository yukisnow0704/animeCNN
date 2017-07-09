animeCNN
===

chainerを用いて、日本のアニメーションのキャラクターを識別するプログラム

## Description

chainerの畳み込みニューラルネットワークを使用してキャラクターを識別するためのデモプログラムです。
動画からキャラクターの画像を出力し、それらを個別に分類することで各キャラクターを下記のように出力することが可能となります。
※GPUを用いた方式は現在、使用することは可能ですが、精度が非常に悪いです。ご注意ください。
また、現在リファクタリングを行なっています。ご了承ください。

## Demo

インプットする画像
![input](https://raw.github.com/wiki/yukisnow/animeCNN/images/input.jpg)

アウトプットされる画像
![input](https://raw.github.com/wiki/yukisnow/animeCNN/images/output.jpg)

## Requirement
 Python 3.6

## Usage

1.動画から画像を出力

python3 video_openCV

parser.add_argument('--cascade_path',               type=str,   default='lbpcascade_animeface.xml')
parser.add_argument('--outputFile',                 type=str,   default='img')
parser.add_argument('--filetype',                   type=str,   default='mp4')
parser.add_argument('--video_path',                 type=str,   default='video')
parser.add_argument('--frameBatch',                 type=int,   default='50')

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

下記のアプリケーションをpipすること
- six
- chainer
- tqdm
- image
