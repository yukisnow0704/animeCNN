# -*- coding: utf-8 -*-

import cv2
import os
import six
import datetime
import pickle
from tqdm import tqdm

import tensorflow as tf

import numpy as np

from bake_model import bake_model
import argparse

def getDataSet(inputPath):
    x_train = []
    y_train = []

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
            y_train.append(i)
    
        print('success read ' + str(fileList[i]))

    f = open('charaName.pickle', 'wb')
    pickle.dump(charaName, f)

    return x_train, y_train, len(charaName)

def train(inputPath, batchNum):
    x_train, y_data, outputNum = getDataSet(inputPath)
    x_train = np.array(x_train).astype(np.float32).reshape(len(x_train), 50, 50, 3) / 255
    y_data = np.array(y_data).astype(np.float32).reshape(len(x_train), 1)

    y_train = np.zeros((len(y_data), int(outputNum)))

    for i in range(len(y_data)):
        y = y_data[i]
        y_train[i][int(y)-1] = 1
    
        # print(y_train[i])
    
    # modelの生成
    
    sess = tf.InteractiveSession()

    bakemodel = bake_model()
    x, y_, y_conv, keep_prob = bakemodel.getModel(outputNum)
    
    saver = tf.train.Saver()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    prediction = y_conv

    sess.run(tf.global_variables_initializer())

    epoch = 1

    while epoch <= 20:
        print("epoch: {}" .format(epoch))
        print(datetime.datetime.now())

        trainImgNum = len(y_train)

        perm = np.random.permutation(trainImgNum)

        pbar = tqdm(total=trainImgNum/batchNum)

        for i in six.moves.range(0, trainImgNum, batchNum):
            x_batch = x_train[perm[i:i+batchNum]]
            y_batch = y_train[perm[i:i+batchNum]]

            x_batch = np.array(x_batch).reshape(len(x_batch), 50, 50, 3)
            y_batch = np.array(y_batch).reshape(len(y_batch), 10)

            sess.run(train_step, feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})

            pbar.update(1)
        
        pbar.close()

        print(y_batch)
        train_accuracy = sess.run(accuracy, feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
        train_prediction = sess.run(prediction, feed_dict={x: x_batch, keep_prob: 1.0})
        print(train_prediction)
        print('step %d, training accuracy %g' % (epoch, train_accuracy))

        epoch += 1
        
    saver.save(sess, 'model/latest_' + str(datetime.datetime.now()) + '/latest_model')
    sess.close()

train('resizeImg', 50)
