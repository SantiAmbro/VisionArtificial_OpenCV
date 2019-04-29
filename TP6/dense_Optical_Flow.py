import cv2 as cv
import numpy as np

cap = cv.VideoCapture("vtest.avi")
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
#Return an array of zeros with the same shape and type as a given array.
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while True:
    ret, frame = cap.read()
    next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #Computes a dense optical flow using the Gunnar Farneback's algorithm.
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #Calculates the magnitude and angle of 2D vectors.
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame', frame)
    cv.imshow('frame2', bgr)

    k = cv.waitKey(30) & 0xff

    if k == ord('q'):
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame)
        cv.imwrite('opticalhsv.png', bgr)

    prvs = next

cap.release()
cv.destroyAllWindows()
