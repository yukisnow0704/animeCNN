# -*- coding:utf-8 -*-

import cv2

from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_path',                 type=str,   default='thumb')
parser.add_argument('--output_path',               type=str,   default='thumbOut')
parser.add_argument('--cascade_path',              type=str,   default='lbpcascade_animeface.xml')
args = parser.parse_args()

imgFileNames = os.listdir(args.file_path)#画像が保存されてるディレクトリへのpath

allImageFile = 0
successFile = 0

for imgFileName in imgFileNames:

    print('now ' + imgFileName)
    
    if not (imgFileName == '.DS_Store'):
        imgFilePath = args.file_path + '/' + imgFileName
        outputFilePath = args.output_path + '/' + imgFileName + '/'

        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)

        if not os.path.exists(outputFilePath):
            os.mkdir(outputFilePath)

        imgNames = os.listdir(imgFilePath)
        
        for imgName in imgNames:
            src_image = cv2.imread(imgFilePath + '/' + imgName)
            cascade=cv2.CascadeClassifier(args.cascade_path)

            facerect = cascade.detectMultiScale(src_image, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

            if len(facerect) > 0:
                count = 0
                for rect in facerect:
                    count += 1
                    croped = src_image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                    cv2.imwrite(os.path.join(outputFilePath, str(count) + '_' + str(imgName) + ".jpg"), croped)
                    successFile += 1

            else:
                pass

            allImageFile += 1

print('openCV face successFile {} / {} allImageFile'.format(successFile, allImageFile))