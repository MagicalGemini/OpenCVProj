# -*- coding: utf-8 -*-

'''
Created on 2016.12.06

@author: FengHao
'''

import sys
import traceback, os

import cv2
from matplotlib import pyplot as plt
from numpy import dtype

import cPickle as pickle
import numpy as np
import base64
import StringIO

MIN_MATCH_COUNT = 10
MIN_KEYPOINT_NUM = 1000
OUTPUT_DEBUG = False
                
def match_sample():
    img1 = cv2.imread('../res/object.png', cv2.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv2.imread('../res/scene5.png', cv2.IMREAD_GRAYSCALE)  # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    #computeKeypoints(img1)

    kp2, des2 = sift.detectAndCompute(img2, None)

    img1 = cv2.drawKeypoints()(img1, kp1, color=(0,255,0), flags=0)
    cv2.imshow("img", img1)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 2, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    plt.imshow(img3, 'gray'), plt.show()


def pickle_keypoints(keypoints, descriptors):
    serialData = []
    kpArray = []
    
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        kpArray.append(temp)

    serialData.append(kpArray)
    serialData.append(descriptors)
    return serialData


def unpickle_keypoints(data):
    keypoints = []

    for point in data[0]:
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        keypoints.append(temp_feature)
        
    descriptors = data[1]
    return keypoints, np.array(descriptors)


def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.getvalue()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)


def computeKeypoints(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if len(keypoints) < MIN_KEYPOINT_NUM:
        print("0|not enough key points")
    else:
        serialData = pickle_keypoints(keypoints, descriptors)
        out = pickle.dumps(serialData, protocol=1)
        strIO = StringIO.StringIO()
        b64IO = StringIO.StringIO()
        strIO.write(out)
        strIO.seek(0)
        base64.encode(strIO, b64IO)
        print("1|" + b64IO.getvalue())


def matchFromSerialData(newImg, serialData):
    
    h, w = newImg.shape[:2]
    if w * h > 1920 * 1080:
        newImg = cv2.resize(newImg, (w / 2, h / 2))
        
    sift = cv2.xfeatures2d.SIFT_create()

    deserialData = pickle.loads(serialData)
    _, des1 = unpickle_keypoints(deserialData)
    _, des2 = sift.detectAndCompute(newImg, None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    
    if len(good) > MIN_MATCH_COUNT:
        print(1)
    else:
        print(0)
    
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("CMD:  compute binaryData")
        print("CMD:  match serialData newBinaryData")
        sys.exit(2)

    cmd = sys.argv[1]
    if cmd == "compute":
                           
        inStrIO = StringIO.StringIO()
        outStrIO = StringIO.StringIO()
        inStrIO.seek(0)
        outStrIO.seek(0)

        if OUTPUT_DEBUG:
            f = open("data.txt", "rb")
            data = f.read()
            f.close()
            inStrIO.write(data)
        else:
            inStrIO.write(sys.argv[2])
        
        base64.decode(inStrIO, outStrIO)
        img = create_opencv_image_from_stringio(outStrIO)

        computeKeypoints(img)
        
    elif cmd == "match":
        serialDataStrIO = StringIO.StringIO()
        serialDataStrIO.seek(0)

        serialDeBase64StrIO = StringIO.StringIO()
        serialDeBase64StrIO.seek(0)

        imgDataIO = StringIO.StringIO()
        imgDataIO.seek(0)

        imgDeBase64IO = StringIO.StringIO()
        imgDeBase64IO.seek(0)

        if OUTPUT_DEBUG:
            f = open("Keypoints", "rb")
            data = f.read()
            f.close()
            serialDataStrIO.write(data)
        else:
            serialDataStrIO.write(sys.argv[2])
        base64.decode(serialDataStrIO, serialDeBase64StrIO)

        imgDataIO.write(sys.argv[3])
        base64.decode(imgDataIO, imgDeBase64IO)

        img = create_opencv_image_from_stringio(imgDeBase64IO)
        
        matchFromSerialData(img, serialDeBase64StrIO.getvalue())
        