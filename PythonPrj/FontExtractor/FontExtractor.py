# -*- coding: utf-8 -*-

'''
Created on 2016.12.28

@author: FengHao

http://jingyan.baidu.com/article/20b68a88be3263796cec62d0.html
http://stackoverflow.com/questions/20670761/letter-inside-letter-pattern-recognition
https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
http://stackoverflow.com/questions/29125349/change-color-of-a-pixel-with-opencv
'''

import cv2

import numpy as np


if __name__ == '__main__':
    
    ImgSRC = cv2.imread("font.jpg")
    imgRGBA = ImgSRC.copy()
    
    imgGray = cv2.cvtColor(imgRGBA, cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.blur(imgGray, (5, 5))

    _, binary = cv2.threshold(imgBlur, 128, 255, cv2.THRESH_BINARY_INV)
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    mor = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    
    _, contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
    maxArea = 0
    maxContour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < maxArea:
            continue
        
        maxArea = area
        maxContour = cnt
    
    mskH = imgRGBA.shape[0]
    mskW = imgRGBA.shape[1]
    mask = np.zeros((mskH, mskW), np.uint8)
        
    if maxContour is not None:
        rect = cv2.minAreaRect(maxContour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        cv2.drawContours(mask, [box], 0, 255, -1)
    
    fontMask = cv2.bitwise_and(binary, binary, mask=mask)
    fontMask = np.int8(fontMask)
      
    imgRGBA = cv2.bitwise_and(imgRGBA, imgRGBA, mask=fontMask)
    imgA8 = np.full((mskH, mskW, 1), 255, np.uint8)
    imgA8 = cv2.bitwise_and(imgA8, imgA8, mask=fontMask)
              
    b, g, r = cv2.split(imgRGBA)
    bgra = [b, g, r, imgA8]
    finalImg = cv2.merge(bgra, 4)

    img_gray = cv2.cvtColor(finalImg,cv2.COLOR_BGR2GRAY)
    new=[[[255%(j + 1), 255%(j + 1), j] for j in i] for i in img_gray]
    dt = np.dtype('f8')
    new=np.array(new,dtype=dt)
    cv2.imwrite("imgRGBA.png", new)    
    
    cv2.waitKey()

