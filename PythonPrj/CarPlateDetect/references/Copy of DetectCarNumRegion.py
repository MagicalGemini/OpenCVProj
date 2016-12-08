'''
Created on 2016-11-27

@author: ponghao
'''

import cv2
import numpy as np

if __name__ == '__main__':

    img = cv2.imread("demo.jpg")
    cv2.imshow("1.Original", img)
    
    # RGB to Gray scale conversion
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow("2.Gray Converted", img_gray)
    
    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    noise_removal = cv2.bilateralFilter(img_gray, 9, 75, 75)
    cv2.imshow("3.Noise Removed", noise_removal)
    
    # Histogram equalisation for better results
    equal_histogram = cv2.equalizeHist(noise_removal)
    cv2.imshow("4.Histogram equalisation", equal_histogram)

    # Morphological opening with a rectangular structure element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations=15)
    cv2.imshow("5.Morphological", morph_image)
    
    # Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
    sub_morp_image = cv2.subtract(equal_histogram, morph_image)
    cv2.imshow("6.Subtraction image", sub_morp_image)

    # Thresholding the image
    ret, thresh_image = cv2.threshold(sub_morp_image, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("7.Image after Thresholding", thresh_image)
    
    # Applying Canny Edge detection
    canny_image = cv2.Canny(thresh_image, 60, 180, apertureSize=3)
    cv2.imshow("8.Image after applying Canny", canny_image)
    
    #canny_image = cv2.convertScaleAbs(canny_image)
    
    # dilation to strengthen the edges
    #kernel = np.ones((3, 3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (7, 7))
    # Creating the kernel for dilation
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
    cv2.imshow("9.Dilation", dilated_image)
    
    # Finding Contours in the image based on edges
    new, contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # Sort the contours based on area ,so that the number plate will be in top 10 contours
    screenCnt = None
    # loop over our contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:  # Select the contour with 4 corners
            screenCnt = approx
            break
        
    final = cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow("10.Image with Selected Contour", final)
    
    # Masking the part other than the number plate
    mask = np.zeros(img_gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1,)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("11.Final_image", new_image)
    
    # Histogram equal for enhancing the number plate for further processing
    y, cr, cb = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2YCrCb))
    # Converting the image to YCrCb model and splitting the 3 channels
    y = cv2.equalizeHist(y)
    # Applying histogram equalisation
    final_image = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)
    # Merging the 3 channels

    cv2.imshow("12.Enhanced Number Plate", final_image)

    cv2.waitKey()  # Wait for a keystroke from the user
