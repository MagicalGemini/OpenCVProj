# -*- coding: utf-8 -*-
'''
Created on 2016-11-27

@author: ponghao

'''

import sys

import cv2

import numpy as np

enableOutput = False
outputPath = "E:/test/"

def filterRect(cnt):    
    rect=cv2.minAreaRect(cnt)  
    box=cv2.boxPoints(rect) 
    box=np.int0(box)  
    angle = rect[2]
    output = False

    w = rect[1][0]
    h = rect[1][1]
    
    if w < h:
        w = rect[1][1]
        h = rect[1][0]
        angle += 90
       
    if w > 0 and h > 0 and abs(int(0-angle)) < 10: 
        area = w * h
        if w / h > 1.2 and w / h < 5 and area > 500 and area < 80000:
#             print("angle", angle)
            output = True
    
    return output

def rmsdiff(im1, im2):
    diff = im1 - im2
    output = False
    if np.sum(abs(diff)) / float(min(np.sum(im1), np.sum(im2))) < 0.1:
        output = True
    return output

def flood_fill_color(img, rects, debugOut = None):

    mskH = img.shape[0] + 2
    mskW = img.shape[1] + 2
    
    final_mask = np.zeros((mskH, mskW), np.uint8)
     
    for rect in rects:
        center = (int(rect[0][0]), int(rect[0][1]))
        w = rect[1][0]
        h = rect[1][0]
   
        if enableOutput:
            cv2.circle(debugOut, center, 1, (0, 255, 0), -1)
            
        minsize = int(min(w, h))
        minsize = int(minsize * 0.2)        
        mask = np.zeros((mskH, mskW), np.uint8)
         
        lodiff = 50
        updiff = 50
        connectivity = 8
        newMaskVal = 255
        numSeeds = 10
        flags = connectivity | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE |  (newMaskVal << 8)
        for i in range(numSeeds):
            x = center[0] + np.random.randint(1000) % minsize - int(minsize / 2)
            y = center[1] + np.random.randint(1000) % minsize - int(minsize / 2)
            seed = (x, y)
            
            if enableOutput:
                cv2.circle(debugOut, seed, 1, (0, 0, 255), -1)
                
            try:
                cv2.floodFill(img, mask, seed, (255, 0, 0), (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags)
            except:
                pass
             
            contours = np.argwhere(mask.transpose() == 255)
            rect = cv2.minAreaRect(contours)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
 
            if filterRect(contours):
                cv2.drawContours(final_mask, [box], 0, 255, -1)
                if enableOutput:
                    color = (0,255, 255)
                    debugOut = cv2.drawContours(debugOut, [box], 0, color, 1)
  
    return final_mask

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Help: CardPlateDetector.exe 'path of inputImg.png'  'path of outputImg.png'")
        sys.exit(2)
    
    inputImg = sys.argv[1]
    outputImg = sys.argv[2]
    
    imgRGBA = cv2.imread(inputImg)

    imgGray = cv2.cvtColor(imgRGBA, cv2.COLOR_RGB2GRAY)
    if enableOutput:
        cv2.imwrite("%s/1-imgGray.png" % outputPath, imgGray)
    
    noise_removal = cv2.bilateralFilter(imgGray, 5, 50, 50)
    if enableOutput:
        cv2.imwrite("%s/2-noise_removal.png" % outputPath, noise_removal)

    sobelx = cv2.Sobel(noise_removal, cv2.CV_8U, 1, 0, ksize=3)
    if enableOutput:
        cv2.imwrite("%s/4-sobelx.png" % outputPath, sobelx)
        
    ret, thresh_image = cv2.threshold(sobelx, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    if enableOutput:
        cv2.imwrite("%s/5-thresh_image.png" % outputPath, thresh_image)
        
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    mor = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, element)
    if enableOutput:
        cv2.imwrite("%s/6-mor.png" % outputPath, mor)
    
    new, contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contoursImg = None
    plateMask = np.zeros(imgGray.shape, np.uint8)

    outConImg = None
    validateRect = []
    if enableOutput:
        outConImg = imgRGBA.copy()
        
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        color = (0, 0, 255)
        if filterRect(cnt):
            color = (0, 255, 0)
            contoursImg = cv2.drawContours(plateMask, [box], 0, 255, -1)
            validateRect.append(rect)
        
        if enableOutput:
            outConImg = cv2.drawContours(outConImg, [box], 0, color, 2)
            
    contoursImg = cv2.bitwise_and(imgRGBA, imgRGBA, mask=plateMask)
    if enableOutput:
        cv2.imwrite("%s/7-contoursImg.png" % outputPath, contoursImg)
    
    finalMsk = flood_fill_color(imgRGBA.copy(), validateRect, outConImg)
    finalMsk = cv2.resize(finalMsk, (imgRGBA.shape[1], imgRGBA.shape[0]))
    if enableOutput:
        cv2.imwrite("%s/8-outConImg.png" % outputPath, outConImg)
        
    h, w = imgRGBA.shape[:2]
    imgA8 = np.full((h, w, 1), 255, np.uint8)
    imgA8 = cv2.bitwise_and(imgA8, imgA8, mask=finalMsk)

    contoursImg = cv2.bitwise_and(imgRGBA, imgRGBA, mask=finalMsk)
    b, g, r = cv2.split(contoursImg)
    bgra = [b, g, r, imgA8]
    finalImg = cv2.merge(bgra, 4)
  
    if enableOutput:
        cv2.imwrite("%s/final.png" % outputPath, finalImg)
                  
    cv2.imwrite(outputImg, finalImg)
    
    cv2.waitKey()
    
