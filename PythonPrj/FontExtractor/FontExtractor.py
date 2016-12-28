# -*- coding: utf-8 -*-

'''
Created on 2016.12.28

@author: FengHao
'''

import cv2
import numpy as np

if __name__ == '__main__':
    
    imgRGBA = cv2.imread("3.jpg")
    cv2.imshow("imgRGBA", imgRGBA)
    
    imgGray = cv2.cvtColor(imgRGBA, cv2.COLOR_RGB2GRAY)
    
    _, imgThresh = cv2.threshold(imgGray, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow("imgThresh", imgThresh)
    
    
#     imgBlur = cv2.blur(imgGray, (3, 3))
#     noise_removal = cv2.bilateralFilter(imgGray, 5, 50, 50)
#     
#     edges = cv2.Canny(noise_removal, 50, 150, apertureSize=3)
#     _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# 
#     cv2.drawContours(imgRGBA, contours, -1, (0, 255, 0), 3)
#     cv2.imshow("imgRGBA", imgRGBA)
#     print("contours size = ", len(contours))
#     mskH = imgRGBA.shape[0] + 2
#     mskW = imgRGBA.shape[1] + 2
#     final_mask = np.zeros((mskH, mskW), np.uint8)
#     
#     for cnt in contours:
#         rect = cv2.minAreaRect(cnt)
# 
#         center = (int(rect[0][0]), int(rect[0][1]))
#         w = rect[1][0]
#         h = rect[1][0]
#                 
#         minsize = int(min(w, h))
#         minsize = int(minsize * 0.5)
#         if minsize == 0:
#             minsize = 1
#             
#         mask = np.zeros((mskH, mskW), np.uint8)
#          
#         lodiff = 50
#         updiff = 50
#         connectivity = 8
#         newMaskVal = 255
#         numSeeds = 10
#         flags = connectivity | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (newMaskVal << 8)
#         for i in range(numSeeds):
#             x = center[0] + np.random.randint(1000) % minsize - int(minsize / 2)
#             y = center[1] + np.random.randint(1000) % minsize - int(minsize / 2)
#             seed = (x, y)
#                  
#             try:
#                 cv2.floodFill(imgRGBA.copy(), mask, seed, (255, 0, 0), (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags)
#             except:
#                 pass
#          
#             cons = np.argwhere(mask.transpose() == 255)
#             rt = cv2.minAreaRect(cons)
#             bx = cv2.boxPoints(rt)
#             bx = np.int0(bx)
#             
#             cv2.drawContours(final_mask, [bx], 0, 255, -1)
#     
#     final_mask = cv2.resize(final_mask, (imgRGBA.shape[1], imgRGBA.shape[0]))
#     h, w = imgRGBA.shape[:2]
#     imgA8 = np.full((h, w, 1), 255, np.uint8)
#     imgA8 = cv2.bitwise_and(imgA8, imgA8, mask=final_mask)
#     
#     contoursImg = cv2.bitwise_and(imgRGBA, imgRGBA, mask=final_mask)
#     b, g, r = cv2.split(contoursImg)
#     bgra = [b, g, r, imgA8]
#     finalImg = cv2.merge(bgra, 4)
#     
#     cv2.imshow("finalImg", finalImg)
    cv2.waitKey()
        
