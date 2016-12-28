# -*- coding: utf-8 -*-

'''
Created on 2016.12.28

@author: FengHao

http://jingyan.baidu.com/article/20b68a88be3263796cec62d0.html
http://stackoverflow.com/questions/20670761/letter-inside-letter-pattern-recognition
https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
'''

import cv2

import numpy as np


def flood_fill_color(img, rects):

    mskH = img.shape[0] + 2
    mskW = img.shape[1] + 2
    
    final_mask = np.zeros((mskH, mskW), np.uint8)
     
    for rect in rects:
        center = (int(rect[0][0]), int(rect[0][1]))
        w = rect[1][0]
        h = rect[1][0]
   
        minsize = int(min(w, h))
        minsize = int(minsize * 0.2)
        mask = np.zeros((mskH, mskW), np.uint8)
        
        lodiff = 50
        updiff = 50
        connectivity = 8
        newMaskVal = 255
        numSeeds = 10
        flags = connectivity | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (newMaskVal << 8)
        for i in range(numSeeds):
            x = center[0] + np.random.randint(1000) % minsize - int(minsize / 2)
            y = center[1] + np.random.randint(1000) % minsize - int(minsize / 2)
            seed = (x, y)
                
            try:
                cv2.floodFill(img, mask, seed, (255, 0, 0), (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags)

            except:
                pass
        
            contours = np.argwhere(mask.transpose() == 255)
            rect = cv2.minAreaRect(contours)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
 
            cv2.drawContours(final_mask, [box], 0, 255, -1)
  
    return final_mask

if __name__ == '__main__':
    
    ImgSRC = cv2.imread("2.jpg")
    imgRGBA = ImgSRC.copy()
    
    imgGray = cv2.cvtColor(imgRGBA, cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.blur(imgGray, (5, 5))

    _, imgThresh = cv2.threshold(imgBlur, 128, 255, cv2.THRESH_BINARY_INV)
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    mor = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, element)
    
    _, contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
    maxArea = 0
    maxContour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < maxArea:
            continue
        
        maxArea = area
        maxContour = cnt
    
    if maxContour is not None:
        rect = cv2.minAreaRect(maxContour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        cv2.drawContours(imgRGBA, [box], -1, (255, 0, 0), 2)
        
        w = rect[1][0]
        h = rect[1][0]
        center = (int(rect[0][0]), int(rect[0][1]))

        lodiff = 80
        updiff = 80
        connectivity = 8
        newMaskVal = 255
        numSeeds = 10
        flags = connectivity | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (newMaskVal << 8)

        mskH = imgRGBA.shape[0] + 2
        mskW = imgRGBA.shape[1] + 2
        mask = np.zeros((mskH, mskW), np.uint8)
        
        minsize = int(min(w, h))
        minsize = int(minsize * 0.3)
        if minsize == 0:
            minsize = 1
            
        for i in range(numSeeds):
            x = center[0] + np.random.randint(1000) % minsize - int(minsize / 2)
            y = center[1] + np.random.randint(1000) % minsize - int(minsize / 2)
            seed = (x, y)
            try:
                cv2.floodFill(imgRGBA, mask, seed, 255, (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags)

            except:
                pass
        
#         cv2.floodFill(imgRGBA, mask, center, 255, (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags);
    
    cv2.imshow("imgRGBA", imgRGBA)        
    cv2.imshow("mask", mask)
    cv2.waitKey()
    cv2.destroyAllWindows()
        
