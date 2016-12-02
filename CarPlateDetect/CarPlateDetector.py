# -*- coding: utf-8 -*-
'''
Created on 2016-11-27

@author: ponghao

'''

import cv2
import numpy as np
import sys

def filterRect(box):    

    output = False

    w = rect[1][0]
    h = rect[1][1]
    
    if w < h:
        w = rect[1][1]
        h = rect[1][0]
       
    if w > 0 and h > 0: 
        area = w * h
        if w / h > 1.2 and w / h < 5 and area > 1000 and area < 80000:
            output = True
    
    return output

def generate_seeds(center, width, height):
    minsize = int(min(width, height))
    minsize = int(minsize - minsize * 0.7)
    
    seed = [None] * 5
    for i in range(5):
        random_integer1 = np.random.randint(1000)
        random_integer2 = np.random.randint(1000)
        seed[i] = (center[0] + random_integer1 % int(minsize / 2) - int(minsize / 2), center[1] + random_integer2 % int(minsize / 2) - int(minsize / 2))
    return seed

def generate_mask(image, seed_point):
    h = image.shape[0]
    w = image.shape[1]
    # OpenCV wants its mask to be exactly two pixels greater than the source image.
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # We choose a color difference of (50,50,50). Thats a guess from my side.
    lodiff = 50
    updiff = 50
    connectivity = 4
    newmaskval = 255
    flags = connectivity + (newmaskval << 8) + cv2.FLOODFILL_FIXED_RANGE + cv2.FLOODFILL_MASK_ONLY
    _ = cv2.floodFill(image, mask, seed_point, (255, 0, 0), (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags)
    return mask

def rmsdiff(im1, im2):
    diff=im1-im2
    output=False
    if np.sum(abs(diff))/float(min(np.sum(im1), np.sum(im2)))<0.3:
        output=True
    return output

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Help: CardPlateDetector.exe 'path of inputImg.png'  'path of outputImg.png'")
        sys.exit(2)
    
    inputImg = sys.argv[1]
    outputImg = sys.argv[2]
    
    imgRGBA = cv2.imread(inputImg)

    imgGray = cv2.cvtColor(imgRGBA, cv2.COLOR_RGB2GRAY)

    imgBlur = cv2.blur(imgGray, (5, 5))

    equal_histogram = cv2.equalizeHist(imgBlur)
    
    sobelx = cv2.Sobel(equal_histogram, cv2.CV_8U, 1, 0, ksize=3)
    
    ret, thresh_image = cv2.threshold(sobelx, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    mor = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, element)
        
    new, contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    contoursImg = None
    plateMask = np.zeros(imgGray.shape, np.uint8)

    validateRect = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if filterRect(box):
            contoursImg = cv2.drawContours(plateMask, [box], 0, 255, -1)
            validateRect.append(rect)
    contoursImg = cv2.bitwise_and(imgRGBA, imgRGBA, mask=plateMask)

    print("validateRect = ", len(validateRect))
    
    mask_list = []
    imgRGBAMask = imgRGBA.copy()
    for rect in validateRect:
        center = (int(rect[0][0]), int(rect[0][1]))
        w = rect[1][0]
        h = rect[1][0]
        seeds = generate_seeds(center, w, h)
         
        for seed in seeds:
            cv2.circle(imgRGBA, seed, 1, (0, 0, 255), -1)
            msk = generate_mask(imgRGBAMask, seed)
 
            contour = np.argwhere(msk.transpose()== 255)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if filterRect(box):
                mask_list.append(msk)
             
    print("mask_list = ", len(mask_list))

    final_masklist=[]
    index=[]
    for i in range(len(mask_list)-1):
        for j in range(i+1, len(mask_list)):
            if rmsdiff(mask_list[i], mask_list[j]):
                index.append(j)
                 
    for mask_no in list(set(range(len(mask_list)))-set(index)):
        final_masklist.append(mask_list[mask_no])
         
    print("final_masklist = ", len(final_masklist))
    
    mskIdx = 0
    idx = outputImg.find(".png")
    subName = outputImg[:idx]
    for msk in final_masklist:
        outputImg = subName +  "%d.png" % mskIdx
        cv2.imwrite(outputImg, msk)
        mskIdx += 1
#     plateMask = np.zeros(imgGray.shape, np.uint8)
#     for msk in final_masklist:
#         contour=np.argwhere(msk.transpose()==255)
#         rect=cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#         plateMask = final_masklist[i]

    h, w = imgRGBA.shape[:2]
    imgA8 = np.full((h, w, 1), 255, np.uint8)
    imgA8 = cv2.bitwise_and(imgA8, imgA8, mask=plateMask)
    b, g, r = cv2.split(contoursImg)
    bgra = [b, g, r, imgA8]
    finalImg = cv2.merge(bgra, 4)
            
    cv2.imwrite(outputImg, finalImg)
    
    cv2.waitKey()
    
