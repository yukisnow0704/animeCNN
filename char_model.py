# -*- coding: utf-8 -*-

import os

import chainer
from chainer import optimizers, cuda
import chainer.functions as F
import chainer.links as L
import chainer.serializers as S

import numpy as np

class clf_bake(chainer.Chain):
    def __init__(self, outputNum):
        super(clf_bake, self).__init__(
            conv1 = L.Convolution2D(3, 16, 2, pad=1),
            conv2 = L.Convolution2D(16, 32, 2, pad=1),
            l3 = L.Linear(None, 1024),
            l4 = L.Linear(None, 256),
            l5 = L.Linear(None, outputNum)
        )
    
    def clear(self):
        self.loss = None
        self.accuracy = None
    
    def forward(self, x_data, y_data, gpu, train=True):
        self.clear()
        with chainer.using_config('train', train):
            if(gpu >= 0):
                x_data = chainer.Variable(cuda.cupy.array(x_data))
                y_data = chainer.Variable(cuda.cupy.array(y_data))
            else:
                x_data = chainer.Variable(np.array(x_data))
                y_data = chainer.Variable(np.array(y_data))

            h = F.max_pooling_2d(F.relu(self.conv1(x_data)), ksize = 5, stride = 2, pad = 2)
            h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize = 5, stride = 2, pad = 2)
            h = F.dropout(F.relu(self.l3(h)))
            h = F.dropout(F.relu(self.l4(h)))
            y = self.l5(h)

            # print (y_data)
            return F.softmax_cross_entropy(y, y_data), F.accuracy(y, y_data)