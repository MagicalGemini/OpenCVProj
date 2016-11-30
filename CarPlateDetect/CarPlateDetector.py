# -*- coding: utf-8 -*-
'''
Created on 2016-11-27

@author: ponghao

'''

import cv2
import numpy as np
import sys

def filterRect(cnt):    
    rect = cv2.minAreaRect(cnt)  
    box = cv2.boxPoints(rect) 
    box = np.int0(box)  
    output = False

    w = rect[1][0]
    h = rect[1][1]
    
    if w > 0 and h > 0 and w * h > 1000:
        output = True
    
    return output

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Help: CardPlateDetector.exe 'path of inputImg.png'  'path of outputImg.png'")
        sys.exit(2)
    
    inputImg = sys.argv[1]
    outputImg = sys.argv[2]
    
    img = cv2.imread(inputImg)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #remove noise
    noise_removal = cv2.bilateralFilter(img_gray, 8, 75, 75)
    
    # Histogram equalisation for better results
    equal_histogram = cv2.equalizeHist(noise_removal)

    sobelx = cv2.Sobel(equal_histogram, cv2.CV_8U, 1, 0, ksize=3)
    
    # Thresholding the image
    ret, thresh_image = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 2))
    
    dilation = cv2.dilate(thresh_image, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    #cv2.imshow("dilation", erosion)
    
#     mor = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, element3)
#     cv2.imshow("morphologyEx1", closing)
    mor = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, element3)
#     cv2.imshow("morphologyEx2", closing)
    
    canny_image = cv2.Canny(mor, 60, 180, apertureSize=3)
    # cv2.imshow("Canny", canny_image)
    
    new, contours, _ = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    contoursImg = None
    failedBox = None
    plateMask = np.zeros(img_gray.shape, np.uint8)
    cpImg = img.copy()

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        if filterRect(cnt):    
            contoursImg = cv2.drawContours(plateMask, [box], 0, 255, -1,)
    
    
    contoursImg = cv2.bitwise_and(img, img, mask=plateMask)

    if contoursImg is not None:
        invertMask = cv2.bitwise_not(plateMask)     
        bg = np.full(img.shape, 255, np.uint8)
        bg = cv2.bitwise_and(bg, bg, mask=invertMask)
        finalImg = cv2.bitwise_or(contoursImg, bg)
        
        h, w = img.shape[:2]
        a = np.full((h, w, 1), 255, np.uint8)
        b, g, r = cv2.split(finalImg)
        bgra = [b, g, r, a]
        finalImg = cv2.merge(bgra, 4)
        cv2.imwrite(outputImg, finalImg)
    
    cv2.waitKey()
    
