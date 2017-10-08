# -*- coding: utf-8 -*-

import cv2
import os
import argparse
from tqdm import tqdm

def detectFace(args, image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)

    cascade = cv2.CascadeClassifier(args.cascade_path)
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

    return facerect

parser = argparse.ArgumentParser()
parser.add_argument('--cascade_path',               type=str,   default='lbpcascade_animeface.xml')
parser.add_argument('--outputFile',                 type=str,   default='img')
parser.add_argument('--filetype',                   type=str,   default='mp4')
parser.add_argument('--video_path',                 type=str,   default='video')
parser.add_argument('--frameBatch',                 type=int,   default='50')
args = parser.parse_args()

count = 0

if not os.path.exists(args.outputFile):
    os.mkdir(args.outputFile)

videos = os.listdir(args.video_path)
pbar = tqdm(videos)

for video in videos:
    if not (video.split(".")[-1] == args.filetype):
        pbar.update(1)
        continue

    if not os.path.exists(args.outputFile + '/' + video):
        os.mkdir(args.outputFile + '/' + video)

    cap = cv2.VideoCapture(args.video_path + '/' + video) #VideoCaptureをcapに保持

    framenum = 0
    faceframenum = 0 #初期化
    color = (255, 255, 255) #白で検出する
    count += 1

    while(cap.isOpened()):
        framenum += 1

        ret, image = cap.read()
        if not ret: # if (ret == False): と同じ
            break

        if framenum%args.frameBatch==0: #frameを50枚ごとに認識する 化物語は動きが少ないからね！！　フレーム数小さいと同じ画像がいっぱいできちゃうよ！
            facerect = detectFace(args, image)
            if len(facerect) == 0: continue #認識結果がnullだったら次のframeへ

            for rect in facerect:
                croped = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                cv2.imwrite(os.path.join(args.outputFile + '/' + video, str(count) + '_' + str(faceframenum) + ".jpg"), croped)
                faceframenum += 1

    cap.release()

pbar.close()
