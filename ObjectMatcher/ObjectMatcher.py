# -*- coding: utf-8 -*-

'''
Created on 2016.08.12

@author: FengHao
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
import cPickle as pickle

MIN_MATCH_COUNT = 10

def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])     
        ++i
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

img1 = cv2.imread('../res/object.png', cv2.IMREAD_GRAYSCALE)  # queryImage
img2 = cv2.imread('../res/scene5.png', cv2.IMREAD_GRAYSCALE)  # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the key points and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)

kps = []
for p in kp1:
    temp = (p.pt, p.size, p.angle, p.response, p.octave, p.class_id)
    kps.append(temp)
bitKps = pickle.dumps(kps)
print(bitKps)

kp2, des2 = sift.detectAndCompute(img2, None)

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
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    
    h, w = img1.shape
    pts = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)
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
