from __future__ import print_function

import cv2 as cv

import numpy as np


# construct the argument parser and parse the arguments
from imutils.video import FPS

cap = cv.VideoCapture("carsRt9_3.avi")
frames_count, fps, width, height = cap.get(cv.CAP_PROP_FRAME_COUNT), cap.get(cv.CAP_PROP_FPS), cap.get(
    cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print(frames_count, fps, width, height)

sub = cv.createBackgroundSubtractorMOG2()  # create background subtractor
# information to start saving a video file
ret, frame = cap.read()  # import image
ratio = 1.0  # resize ratio
image = cv.resize(frame, (0, 0), None, ratio, ratio)  # resize image
width2, height2, channels = image.shape

fps = None

while True:
    ret, frame = cap.read()
    # initialize the FPS throughput estimator
    fps = FPS().start()

    image = cv.resize(frame, (0, 0), None, ratio, ratio)  # resize image
    cv.imshow("image", image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # converts image to gray
    cv.imshow("gray", gray)
    fgmask = sub.apply(gray)  # uses the background subtraction
    cv.imshow("fgmask", fgmask)
    # applies different thresholds to fgmask to try and isolate cars
    # just have to keep playing around with settings until cars are easily identifiable
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
    closing = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
    cv.imshow("closing", closing)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    cv.imshow("opening", opening)
    dilation = cv.dilate(opening, kernel)
    cv.imshow("dilation", dilation)
    retvalbin, bins = cv.threshold(dilation, 220, 255, cv.THRESH_BINARY)  # removes the shadows
    cv.imshow("retvalbin", retvalbin)
    # creates contours
    contours, hierarchy = cv.findContours(bins, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    minarea = 400
    # max area for contours, can be quite large for buses
    maxarea = 50000
    # vectors for the x and y locations of contour centroids in current frame
    cxx = np.zeros(len(contours))
    cyy = np.zeros(len(contours))


    for i in range(len(contours)):  # cycles through all contours in current frame
        if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
            area = cv.contourArea(contours[i])  # area of contour
            if minarea < area < maxarea:  # area threshold for contour
                # calculating centroids of contours
                cnt = contours[i]
                M = cv.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # gets bounding points of contour to create rectangle
                # x,y is top left corner and w,h is width and height
                x, y, w, h = cv.boundingRect(cnt)

                roi = frame[y:y + h, x:x + w]
                cv.imshow("cars", roi)
                # creates a rectangle around contour
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Prints centroid text in order to double check later on
                # cv.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv.FONT_HERSHEY_SIMPLEX, .3,
                #            (0, 0, 255), 1)
                cv.drawMarker(image, (cx, cy), (0, 255, 255), cv.MARKER_CROSS, markerSize=8, thickness=3,
                              line_type=cv.LINE_8)
                # update the FPS counter
                fps.update()
                fps.stop()

                cv.putText(image, "{:.2f}".format(fps.fps()) + ' fps', (cx + 10, cy + 10), cv.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)

    cv.imshow("countours", image)
    key = cv.waitKey(20)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
