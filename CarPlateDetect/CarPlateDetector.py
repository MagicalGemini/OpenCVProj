'''
Created on 2016-11-27

@author: ponghao

@reference: http://nbviewer.jupyter.org/gist/kislayabhi/89b985e5b78a6f56029a
'''

import cv2
import numpy as np

def validate(cnt):    
    rect = cv2.minAreaRect(cnt)  
    box = cv2.boxPoints(rect) 
    box = np.int0(box)  
    output = False

    w = rect[1][0]
    h = rect[1][1]
                
    if  (h * w < 16000) and (h * w > 2000) and (w > h and h > 0 and w / h > 3):
        print("w=%d, h=%d" % (w, h))
        output = True
        
    return output

def generate_seeds(centre, width, height):
    minsize = int(min(width, height))
    seed = [None] * 10
    for i in range(10):
        random_integer1 = np.random.randint(1000)
        random_integer2 = np.random.randint(1000)
        seed[i] = (centre[0] + random_integer1 % int(minsize / 2) - int(minsize / 2), centre[1] + random_integer2 % int(minsize / 2) - int(minsize / 2))
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
    cv2.floodFill(image, mask, seed_point, (255, 0, 0), (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags)
    return mask

if __name__ == '__main__':

    img = cv2.imread("demo.jpg")
    #img = cv2.imread("car8.jpg")

    cv2.imshow("OriImg", img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    noise_removal = cv2.bilateralFilter(img_gray, 9, 75, 75)
    #cv2.imshow("noise_removal", noise_removal)
    
    #noise_removal = cv2.GaussianBlur(img_gray, (5, 5), 0)
    #cv2.imshow("noise_removal1", noise_removal)
    
    # Histogram equalisation for better results
    equal_histogram = cv2.equalizeHist(noise_removal)

    sobelx = cv2.Sobel(equal_histogram, cv2.CV_8U, 1, 0, ksize=3)
    
    # Thresholding the image
    ret, thresh_image = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("Thresholding", thresh_image)
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 2))
    closing = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, element)
    # cv2.imshow("morphologyEx", closing)
    
    canny_image = cv2.Canny(closing, 60, 180, apertureSize=3)
    # cv2.imshow("Canny", canny_image)
    
    # new, contours, _ = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    new, contours, _ = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    new_image = None
    plateMask = np.zeros(img_gray.shape, np.uint8)
    for cnt in contours:
        if validate(cnt):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #conImg = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
             
            new_image = cv2.drawContours(plateMask, [box], 0, 255, -1,)
            new_image = cv2.bitwise_and(img, img, mask=plateMask)
            y, cr, cb = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2YCrCb))
            y = cv2.equalizeHist(y)
    
    final_image = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)
    cv2.imshow("final_img", final_image)
    
    cv2.waitKey()  # Wait for a keystroke from the user
