# -*- coding: utf-8 -*-

import numpy as np
import six
import cv2
import os

import chainer
from chainer import computational_graph as c
import chainer.functions as F
import chainer.serializers as S
from chainer import optimizers, cuda

from char_model import clf_bake

import pickle
import argparse

def forward(x_data, gpu):

    with chainer.using_config('train', False):
        if(gpu >= 0):
            x = chainer.Variable(cuda.cupy.array(x_data))
        else:
            x = chainer.Variable(np.array(x_data))

        h = F.max_pooling_2d(F.relu(model.conv1(x)), ksize = 5, stride = 2, pad =2)
        h = F.max_pooling_2d(F.relu(model.conv2(h)), ksize = 5, stride = 2, pad =2)
        h = F.dropout(F.relu(model.l3(h)))
        h = F.dropout(F.relu(model.l4(h)))
        y = model.l5(h)

    return y

def detect(image, cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    
    cascade = cv2.CascadeClassifier(cascade_file)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(image, scaleFactor = 1.1, minNeighbors = 1, minSize = (24, 24))

    return faces

def recognition(image, faces):
    face_images = []

    for (x, y, w, h) in faces:
        dst = image[y:y+h, x:x+w]
        dst = cv2.resize(dst, (50, 50))
        face_images.append(dst)
    
    face_images = np.array(face_images).astype(np.float32).reshape(len(face_images), 3, 50, 50) /255

    return forward(face_images, gpu), image

def draw_result(image, faces, result):
    count = 0
    for (x, y, w, h) in faces:
        result_data = result.data[count]
        classNum = result_data.argmax()
        
        print(classNum)
        recognized_class = chara_name[int(classNum)]
        if(recognized_class == 'other'):
            cv2.rectangle(image, (x,y), (x+w, y+h), (255, 140, 0), 3)
            cv2.putText(image, recognized_class, (x+5, y+h-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 140, 0), 1, 16)
        else:
            cv2.rectangle(image, (x,y), (x+w, y+h), (0, 140, 255), 3)
            cv2.putText(image, recognized_class, (x+5, y+h-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 140, 255), 1, 16)

        count += 1
    
    return image


parser = argparse.ArgumentParser()
parser.add_argument('--model_path',                 type=str,   default='model/train_latest.pickle')
parser.add_argument('--charaName_path',             type=str,   default='charaName.pickle')
parser.add_argument('--input_path',                 type=str,   default='input')
parser.add_argument('--output_path',                 type=str,   default='output')
parser.add_argument('--gpu',                        type=int,   default='0')
args = parser.parse_args()

f = open(args.model_path, 'rb')
model = pickle.load(f)

f = open(args.charaName_path, 'rb')
chara_name = pickle.load(f)

inputPath = args.input_path
outputPath = args.output_path
gpu = args.gpu

fileList = os.listdir(inputPath)

for fileName in fileList:
    if (fileName == '.DS_Store'):continue
    if (fileName == '._.DS_Store'):continue

    img = cv2.imread(inputPath + '/' + fileName)
    # cv2.imshow("loaded", img)
    faces = detect(img)

    print(fileName)

    result, image = recognition(img, faces)

    image = draw_result(image, faces, result)
    cv2.imwrite(outputPath + '/output-' + fileName, image)