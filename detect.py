import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def detect_empty(image_name):
    img = cv2.imread(filename=image_name)
    #to make the image grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    #apply gaussian filter
    blur_gray = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=2)
    #otsu thresholds
    high_thresh, thresh_im = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    #canny edge detector
    edges = cv2.Canny(image=blur_gray, threshold1=lowThresh, threshold2=high_thresh, apertureSize=3)
    
    ### Hough Lines
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img[:,:,0]) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    #binarize the image
    ret, bin_ = cv2.threshold(line_image,0,1,cv2.THRESH_BINARY_INV)
    
    #Erosion
    kernel = np.ones((5,5), np.uint8)
    bin_dilation = cv2.erode(bin_, kernel, iterations=6)
    
    _, contours, _ = cv2.findContours(bin_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boundRect = [None]*len(contours)
    contours_poly = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    
    
    min_width = 10
    min_height = 10

    selected_rectangles = []
    for rec in boundRect:
        if rec[2] >= min_width and rec[3] >= min_height:
            if rec[0] > 0 and rec[1] > 0:
                selected_rectangles.append(rec)
                
    output = []
    for i in range(len(selected_rectangles)):
        output.append([i+1, selected_rectangles[i][0], selected_rectangles[i][1], selected_rectangles[i][0]+selected_rectangles[i][2], selected_rectangles[i][1]+selected_rectangles[i][3]])
    print(output)
    return output


if __name__=="__main__":
    PATH = "shelf-1.jpg"
    output = detect_empty(PATH)
