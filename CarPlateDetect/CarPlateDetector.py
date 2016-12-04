# -*- coding: utf-8 -*-
'''
Created on 2016-11-27

@author: ponghao

http://blog.csdn.net/jinshengtao/article/details/17883075/#
http://nbviewer.jupyter.org/gist/kislayabhi/89b985e5b78a6f56029a
http://blog.csdn.net/poem_qianmo/article/details/28261997
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
        if w / h > 1.2 and w / h < 5 and area > 1000 and area < 80000:
#             print("angle", angle)
            output = True
    
    return output

def rmsdiff(im1, im2):
    diff = im1 - im2
    output = False
    if np.sum(abs(diff)) / float(min(np.sum(im1), np.sum(im2))) < 0.1:
        output = True
    return output

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
    
    noise_removal = cv2.bilateralFilter(imgGray, -1, 50, 50)
    if enableOutput:
        cv2.imwrite("%s/2-noise_removal.png" % outputPath, noise_removal)
                
#     equal_histogram = cv2.equalizeHist(noise_removal)
#     if enableOutput:
#         cv2.imwrite("%s/3-equal_histogram.png" % outputPath, equal_histogram)
        
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

    if enableOutput:
        outConImg = imgRGBA.copy()
    validateRect = []
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
    
#     print("validateRect = ", len(validateRect))
#
#     mask_list = []
#     imgRGBAMask = imgRGBA.copy()
#     mskH = imgRGBAMask.shape[0] + 2
#     mskW = imgRGBAMask.shape[1] + 2
#     tmpMsk = plateMask.copy()
#     tmpMsk = cv2.resize(tmpMsk, (mskW, mskH))
#     if enableOutput:
#         cv2.imwrite("%s/tmpMsk.png" % outputPath, tmpMsk)
#     for rect in validateRect:
#         center = (int(rect[0][0]), int(rect[0][1]))
#         w = rect[1][0]
#         h = rect[1][0]
#   
#         cv2.circle(outConImg, center, 1, (0, 255, 0), -1)
#         minsize = int(min(w, h))
#         minsize = int(minsize * 0.3)        
#         mask = np.zeros((mskH, mskW), np.uint8)
#         
#         lodiff = 50
#         updiff = 50
#         connectivity = 8
#         newMaskVal = 255
#         numSeeds = 5
#         flags = connectivity | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE |  (newMaskVal << 8)
#         for i in range(numSeeds):
#             x = center[0] + np.random.randint(1000) % minsize - int(minsize / 2)
#             y = center[1] + np.random.randint(1000) % minsize - int(minsize / 2)
#             seed = (x, y)
#             cv2.circle(outConImg, seed, 1, (0, 0, 255), -1)
#             try:
#                 fillRect = cv2.floodFill(imgRGBAMask, mask, seed, (255, 0, 0), (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags)
#             except:
#                 pass
#             
#             contours = np.argwhere(mask.transpose() == 255)
#             rect = cv2.minAreaRect(contours)
#             box = cv2.boxPoints(rect)
#             box = np.int0(box)
# 
#             if filterRect(contours):
#                 if enableOutput:
#                     color = (0,255, 255)
#                     outConImg = cv2.drawContours(outConImg, [box], 0, color, 1)
# #             else:
# #                 if enableOutput:
# #                     color = (255, 0, 0)
# #                     outConImg = cv2.drawContours(outConImg, [box], 0, color, 1)
#                 #mask = cv2.bitwise_and(mask, mask, mask=tmpMsk)
#                 mask_list.append(mask)
#      
#     if enableOutput:
#         cv2.imwrite("%s/8-outConImg.png" % outputPath, outConImg)
#                               
#     print("mask_list = ", len(mask_list))
# 
#     final_masklist = []
#     index = []
#     for i in range(len(mask_list) - 1):
#         for j in range(i + 1, len(mask_list)):
#             if rmsdiff(mask_list[i], mask_list[j]):
#                 index.append(j)
#                 
# #     for i in range(len(mask_list)):
# #         hasSameMsk = False
# #         for j in range(i + 1, len(mask_list)):
# #             if rmsdiff(mask_list[i], mask_list[j]):
#                 
#             
#     for mask_no in list(set(range(len(mask_list))) - set(index)):
#         final_masklist.append(mask_list[mask_no])
#              
#     print("final_masklist = ", len(final_masklist))
#  
#     mskIdx = 0
#     idx = outputImg.find(".png")
#     subName = outputImg[:idx]
#     for msk in final_masklist:
#         if enableOutput:
#             out = "%s/final_msk_%d.png" % (outputPath, mskIdx)
#             cv2.imwrite(out, msk)
#             
#             #mask = cv2.bitwise_or(mask, tmpMsk)
#             out = "%s/final_and_msk_%d.png" % (outputPath, mskIdx)
#             cv2.imwrite(out, msk)
#         mskIdx += 1

    h, w = imgRGBA.shape[:2]
    imgA8 = np.full((h, w, 1), 255, np.uint8)
    imgA8 = cv2.bitwise_and(imgA8, imgA8, mask=plateMask)
    b, g, r = cv2.split(contoursImg)
    bgra = [b, g, r, imgA8]
    finalImg = cv2.merge(bgra, 4)
            
    cv2.imwrite(outputImg, finalImg)
    
    cv2.waitKey()
    
