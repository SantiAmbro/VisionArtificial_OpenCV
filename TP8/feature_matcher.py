import cv2 as cv
import argparse

import numpy as np


def nothing(x):
    pass


parser = argparse.ArgumentParser()
parser.add_argument('--facedet', type=str, help='Feature Detection method ORB.', default='ORB')
parser.add_argument('--matcher', type=str, help='Matcher method', default='BRUTE')
parser.add_argument('--image', type=str, help='Name of image', default='')
args = parser.parse_args()

cv.namedWindow("Webcam")

capture = cv.VideoCapture(0)

orb = cv.ORB_create(nfeatures=1500)

time = 0
img_counter = 0
image = None
firstImage = True
alreadyProcessed = False

while True:
    if not img_counter == 0:
        image = cv.imread(args.image)

    ret, frame = capture.read()
    if frame is None:
        break

    t1 = cv.getTickCount()
    kp2, des2 = orb.detectAndCompute(frame, None)
    frame = cv.drawKeypoints(frame, kp2, None)
    if not img_counter == 0:
        kp1, des1 = orb.detectAndCompute(image, None)
        if not alreadyProcessed:
            image = cv.drawKeypoints(image, kp1, None)
        if args.matcher == 'BRUTE':
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            matching_result = cv.drawMatches(image, kp1, frame, kp2, matches[:50], None, flags=2)
            cv.imshow('Result', matching_result)
        alreadyProcessed = True
    t2 = cv.getTickCount()
    time = (t2 - t1) / cv.getTickFrequency()

    cv.rectangle(frame, (10, 2), (250, 20), (255, 255, 255), -1)
    cv.putText(frame, str(args.facedet) + str(" ") + str(time) + str(" Seconds"), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.rectangle(frame, (10, 30), (200, 50), (255, 255, 255), -1)
    cv.putText(frame, str(args.matcher) + str(" ") + str("Matcher"), (45, 45),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Webcam', frame)
    if firstImage and not img_counter == 0:
        cv.imshow('Image', image)
        firstImage = False

    keyboard = cv.waitKey(30)
    if keyboard == ord('b'):
        args.matcher = 'BRUTE'
    elif keyboard == ord('n'):
        args.matcher = 'FLANN'
    elif keyboard == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv.imwrite(img_name, frame)
        args.image = img_name
        print("{} written!".format(img_name))
        img_counter += 1
    elif keyboard == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
