import tensorflow as tf
import pickle

import os
import cv2
import six
import numpy as np

from bake_model import bake_model

# modelの呼び出し
def getModel(modelFilePath = 'model/latest_2017-10-05 05:46:27.420873/latest_model'):
    
    # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    # accuracy = None
    prediction = tf.argmax(y_conv,1)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    saver.restore(sess, modelFilePath)

    return sess, prediction

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
    
    face_images = np.array(face_images).astype(np.float32).reshape(len(face_images), 50, 50, 3) /255

    return forward(face_images)

def forward(x_data):

    # y = sess.run(accuracy, feed_dict={x: x_data, keep_prob: 1.0})
    y = accuracy.eval(feed_dict={x: x_data, keep_prob: 1.0}, session=sess)
    print(y)

    return y

def draw_result(image, faces, result):
    count = 0
    for (x, y, w, h) in faces:
        classNum = result[count]
        
        recognized_class = chara_name[int(classNum)]
        if(recognized_class == 'other'):
            cv2.rectangle(image, (x,y), (x+w, y+h), (255, 140, 0), 3)
            cv2.putText(image, recognized_class, (x+5, y+h-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 140, 0), 1, 16)
        else:
            cv2.rectangle(image, (x,y), (x+w, y+h), (0, 140, 255), 3)
            cv2.putText(image, recognized_class, (x+5, y+h-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 140, 255), 1, 16)

        count += 1
    
    return image


# 設定
inputPath = 'input'

# ファイルの呼び出し
f = open('charaName.pickle', 'rb')
chara_name = pickle.load(f)

# modelの取得
bakemodel = bake_model()
x, y_, y_conv, keep_prob = bakemodel.getModel(10)

sess, accuracy = getModel()

# 調査したい画像データの読み込み
fileList = os.listdir(inputPath)

for fileName in fileList:
    if (fileName == '.DS_Store'):continue
    if (fileName == '._.DS_Store'):continue

    img = cv2.imread(inputPath + '/' + fileName)
    cv2.imshow("loaded", img)

    faces = detect(img)

    result = recognition(img, faces)
    
    image = draw_result(img, faces, result)
    cv2.imwrite('output' + '/output-' + fileName, image)
