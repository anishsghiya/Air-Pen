# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:04:36 2020

@author: ANISH
"""

import cv2
import numpy as np 
import time 

kernel = np.ones((5,5),np.uint8)
noiseth = 500
x1,y1=0,0
canvas = None

# Load these 2 images and resize them to the same size.
pen_img = cv2.resize(cv2.imread('pen.png',1), (100, 100))
eraser_img = cv2.resize(cv2.imread('eraser.png',1), (100, 100))

# Create a background subtractor Object
backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

# This threshold determines the amount of disruption in the background.
background_threshold = 600

# With this variable we will monitor the time between previous switch.
last_switch = time.time()

# A variable which tells you if you're using a pen or an eraser.
switch = 'Pen'

def nothing(x):
    pass

# Initializing the webcam feed.
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Create a window named trackbars.
#cv2.namedWindow("Trackbars")

# Now create 6 trackbars that will control the lower and upper range of 
# H,S and V channels. The Arguments are like this: Name of trackbar, 
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.

 
while True:
    
    # Start reading the webcam feed frame by frame.
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally (Not required)
    frame = cv2.flip( frame, 1 ) 
    
    top_left = frame[0: 50, 0: 50]
    fgmask = backgroundobject.apply(top_left)
    
    # Note the number of pixels that are white, this is the level of 
    # disruption.
    switch_thresh = np.sum(fgmask==255)
    
    # If the disruption is greater than background threshold and there has 
    # been some time after the previous switch then you. can change the 
    # object type.
    if switch_thresh>background_threshold and (time.time()-last_switch) > 1:

        # Save the time of the switch. 
        last_switch = time.time()
        
        if switch == 'Pen':
            switch = 'Eraser'
        else:
            switch = 'Pen'


    # Convert the BGR image to HSV image.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
    # Set the lower and upper HSV range according to the value selected
    # by the trackbar
    lower_range = np.array([88, 108, 76])
    upper_range = np.array([162, 255, 255])
    
    # Filter the image and get the binary mask, where white represents 
    # your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)
 
    # You can also visualize the real part of the target color (Optional)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Converting the binary mask to 3 channel image, this is just so 
    # we can stack it with the others
    #mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # stack the mask, orginal frame and the filtered result
    
    # Perform the morphological operations to get rid of the noise.
    # Erosion Eats away the white part while dilation expands it.
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Find Contours in the frame.
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    # Make sure there is a contour present and also make sure its size 
    # is bigger than noise threshold.
    if canvas is None:
        canvas = np.zeros_like(frame)
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:
        
        # Grab the biggest contour with respect to area
        c = max(contours, key = cv2.contourArea)
        
        # Get bounding box coordinates around that contour
        x2,y2,w,h = cv2.boundingRect(c)
        
        # Draw that bounding box
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,25,255),2) 
    
        # If there were no previous points then save the detected x2,y2 
        # coordinates as x1,y1. 
        # This is true when we writing for the first time or when writing 
        # again when the pen had disappeared from view.
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2
            
        else:
            if switch == 'Pen':
                # Draw the line on the canvas
                canvas = cv2.line(canvas, (x1,y1),(x2+ (w//2),y2 + (h//2)), [255,0,0], 5)
                
            else:
                cv2.circle(canvas, (x2 + (w//2), y2 + (h//2)), 20,(0,0,0), -1)
            #else:
            # Draw the line on the canvas
            #canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 4)
        
        # After the line is drawn the new points become the previous points.
        x1,y1= x2 + (w//2), y2 + (h//2)

    else:
        # If there were no contours detected then make x1,y1 = 0
        x1,y1 =0,0
    
    # Now this piece of code is just for smooth drawing. (Optional)
    _ , mask = cv2.threshold(cv2.cvtColor (canvas, cv2.COLOR_BGR2GRAY), 20, 
    255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
    background = cv2.bitwise_and(frame, frame,
    mask = cv2.bitwise_not(mask))
    frame = cv2.add(foreground,background)
    
    # Switch the images depending upon what we're using, pen or eraser.
    if switch != 'Pen':
        cv2.circle(frame, (x1, y1), 20, (255,255,255), -1)
        frame[0: 100, 0: 100] = eraser_img
    else:
        frame[0: 100, 0: 100] = pen_img

    #cv2.imshow('image',frame)
    
    # Merge the canvas and the frame.
    #frame = cv2.add(frame,canvas)
    
    # Optionally stack both frames and show it.
    stacked = np.hstack((canvas,frame))
    cv2.imshow('Air Pen',cv2.resize(stacked,None,fx=0.6,fy=0.6))

    #cv2.imshow('image',frame)
    
    #stacked = np.hstack((mask_3,frame,res))
    # Show this stacked frame at 40% of the size.
    #cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
    
    # If the user presses ESC then exit the program
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    # When c is pressed clear the canvas
    if key == ord('c'):
        canvas = None
        
    # If the user presses `s` then print this array.
    """elif key == ord('s'):
        
        thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
        print(thearray)
        
        # Also save this array as penval.npy
        np.save('penval',thearray)
        break
    """
    
# Release the camera & destroy the windows.    
cap.release()
cv2.destroyAllWindows()