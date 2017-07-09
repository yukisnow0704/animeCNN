# -*- coding: utf-8 -*-

import cv2
import os
import six
import datetime
from tqdm import tqdm

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
from char_model import clf_bake

import pickle
import numpy as np

def getDataSet(inputPath):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    charaName = []

    fileList = os.listdir(inputPath)
    
    fileListNum = len(fileList)

    for i in range(fileListNum):

        if (fileList[i] == '.DS_Store'):continue
        if (fileList[i] == '._.DS_Store'):continue

        imgList = os.listdir(inputPath + '/' + fileList[i])

        charaName.append(fileList[i])

        imgNum = len(imgList)
        
        for j in range(imgNum):
            imgSrc = cv2.imread(inputPath + '/' + fileList[i] + "/" + imgList[j])

            if imgSrc is None:continue

            x_train.append(imgSrc)
            y_train.append(i-1)
    
        print('success read {}',format(fileList[i]))

    f = open('charaName.pickle', 'wb')
    pickle.dump(charaName, f)

    return x_train, y_train, x_test, y_test, len(charaName)

def train(inputPath, outputModelPath, epochNum, batchNum, gpu):
    x_train, y_train, x_test, y_test, outputNum = getDataSet(inputPath)
    x_train = np.array(x_train).astype(np.float32).reshape(len(x_train), 3, 50, 50) / 255
    y_train = np.array(y_train).astype(np.int32)
    x_test = np.array(x_test).astype(np.float32).reshape(len(x_test), 3, 50, 50) / 255
    y_test = np.array(y_test).astype(np.int32)
    
    model = clf_bake(outputNum)

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    epoch = 1

    while epoch <= epochNum:
        print("epoch: {}" .format(epoch))
        print(datetime.datetime.now())

        trainImgNum = len(y_train)
        testImgNum = len(y_test)

        sumAcr = 0
        sumLoss = 0

        perm = np.random.permutation(trainImgNum)

        pbar = tqdm(total=trainImgNum/batchNum)

        for i in six.moves.range(0, trainImgNum, batchNum):
            x_batch = x_train[perm[i:i+batchNum]]
            y_batch = y_train[perm[i:i+batchNum]]

            # optimizer.zero_grads()
            loss, acc = model.forward(x_batch, y_batch, gpu)
            loss.backward()

            # optimizer.update()

            sumLoss += float(loss.data) * len(y_batch)
            sumAcr += float(acc.data) * len(y_batch)
            pbar.update(1)
        
        pbar.close()
        print('train mean loss={}, accuracy={}'.format(sumLoss / trainImgNum, sumAcr / trainImgNum))

        epoch += 1
        
        # f = open(outputModelPath+'/train'+ '_' + str(datetime.datetime.now()) +'.pickle', 'wb')
        # pickle.dump(model, f)
    
    f = open(outputModelPath+'/train_latest.pickle', 'wb')
    pickle.dump(model, f)

gpu = -1
epochNum = 50
batchNum = 50
inputPath = "resizeImg"
outputModelPath = 'model'
train(inputPath, outputModelPath, epochNum, batchNum, gpu)