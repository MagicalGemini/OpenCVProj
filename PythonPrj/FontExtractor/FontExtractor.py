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

if __name__ == '__main__':
    
    ImgSRC = cv2.imread("3.jpg")
    imgRGBA = ImgSRC.copy()
    
    imgGray = cv2.cvtColor(imgRGBA, cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.blur(imgGray, (5, 5))

    _, imgThresh = cv2.threshold(imgBlur, 128, 255, cv2.THRESH_BINARY_INV)
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    mor = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, element)
    
    _, contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    print("contours size = ", len(contours))
    
#     mskH = ImgSRC.shape[0] + 2
#     mskW = ImgSRC.shape[1] + 2
#     mask = np.zeros((mskH, mskW), np.uint8)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
#         w = rect[1][0]
#         h = rect[1][0]
#         center = (int(rect[0][0]), int(rect[0][1]))
#         start = (int(rect[0][0] - h / 2), int(rect[0][1] - w / 2))
#         
        box = cv2.boxPoints(rect)
        box = np.int0(box)
#         
#         lodiff = 80
#         updiff = 80
#         connectivity = 8
#         newMaskVal = 255
#         flags = connectivity | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (newMaskVal << 8)
#         
#         cv2.circle(imgRGBA, center, 1, (0, 0, 255), 3)
#         cv2.floodFill(imgRGBA, mask, center, 255);
#         cv2.floodFill(imgRGBA, mask, center, 255, (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags);
        cv2.drawContours(imgRGBA, [box], -1, (255, 0, 0), 2)
    
            
    cv2.imshow("imgRGBA", imgRGBA)
    cv2.waitKey()
    cv2.destroyAllWindows()
        
