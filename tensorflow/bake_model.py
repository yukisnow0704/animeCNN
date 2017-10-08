# -*- coding: utf-8 -*-
import tensorflow as tf

class bake_model():
    def __init__(self):
        pass

    # weight_variable
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # bias_variable
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # モデルの作成
    def getModel(self, outputNum):

        x = tf.placeholder(tf.float32, shape=[None, 50, 50, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        
        # convを用いたCNNを実行
        # conv reshape
        x_image = tf.reshape(x, [-1, 50, 50, 3])

        # conv1層目作成
        W_conv1 = self.weight_variable([5, 5, 3, 32])
        b_conv1 = self.bias_variable([32])
        h_conf1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conf1)

        # conv2層目作成
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conf2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conf2)

        # 隠れ層1の作成
        W_fc1 = self.weight_variable([13 * 13 * 64, 4096])
        b_fc1 = self.bias_variable([4096])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 13 * 13 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # 隠れ層1の作成
        W_fc2 = self.weight_variable([4096, 256])
        b_fc2 = self.bias_variable([256])
        # h_pool3_flat = tf.reshape(h_pool2, [-1, 4096])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        # DropOut層を作成
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc2, keep_prob)

        # 出力層を作成
        W_fc3 = self.weight_variable([256, 10])
        b_fc3 = self.bias_variable([10])

        # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)
        y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

        return x, y_, y_conv, keep_prob