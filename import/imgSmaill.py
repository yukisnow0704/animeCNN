# -*- coding:utf-8 -*-

from PIL import Image
import os
import argparse

def readImg(imgFilePath, imgName):
    try:
        img_src = Image.open(imgFilePath + '/' + imgName)
        print("read img!")
    except:
        print("{} is not image file!".format(imgName))
        img_src = 1
    return img_src

parser = argparse.ArgumentParser()
parser.add_argument('--file_path',                 type=str,   default='thumb')
parser.add_argument('--output_path',               type=str,   default='thumbOut')
args = parser.parse_args()

imgFileNames = os.listdir(args.file_path)#画像が保存されてるディレクトリへのpath

for imgFileName in imgFileNames:
    if not (imgFileName == '.DS_Store'):
        imgFilePath = args.file_path + '/' + imgFileName
        outputFilePath = args.output_path + '/' + imgFileName + '/'
        
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)

        if not os.path.exists(outputFilePath):
            os.mkdir(outputFilePath)

        imgNames = os.listdir(imgFilePath)#画像が保存されてるディレクトリへのpath
        
        for imgName in imgNames:
            img_src = readImg(imgFilePath, imgName)
            if img_src == 1:continue
            else:
                resizedImg = img_src.resize((50,50)) #
                resizedImg.save(outputFilePath + "50_50_" + imgName)#名前は長くなっちゃうけど仕方ない。
                #上下反転
                tmp = img_src.transpose(Image.FLIP_TOP_BOTTOM)
                tmp.save(outputFilePath + "flipTB_50_50" + imgName)
                #90度回転
                tmp = img_src.transpose(Image.ROTATE_90)
                tmp.save(outputFilePath + "spin90_50_50" + imgName)
                #270度回転
                tmp = img_src.transpose(Image.ROTATE_270)
                tmp.save(outputFilePath + "spin270_50_50" + imgName)
                #左右反転
                tmp = img_src.transpose(Image.FLIP_LEFT_RIGHT)
                tmp.save(outputFilePath + "flipLR_50_50" + imgName)
                print(imgName+" is done!")