import cv2 as cv
import numpy as np
import argparse


def nothing(x):
    pass


parser = argparse.ArgumentParser()
parser.add_argument('--minTh', type=str, help='Min Threshold', default=10)
parser.add_argument('--maxTh', type=str, help='Max Threshold', default=200)
args = parser.parse_args()

cv.namedWindow("Webcam")

# Setup SimpleBlobDetector parameters.
params = cv.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

capture = cv.VideoCapture(0)

time = 0

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # Setup Threshold
    params.minThreshold = args.minTh
    params.maxThreshold = args.maxTh
    # Create Blob Detector
    detector = cv.SimpleBlobDetector_create(params)
    minth = cv.getTrackbarPos("Min Threshold: ", "Webcam")
    maxth = cv.getTrackbarPos("Max Threshold: ", "Webcam")
    cv.createTrackbar("Min Threshold: ", "Webcam", 10, 1000, nothing)
    cv.createTrackbar("Max Threshold: ", "Webcam", 200, 1000, nothing)

    t1 = cv.getTickCount()
    args.minTh = minth
    args.maxTh = maxth
    keypoints = detector.detect(frame)
    frame = cv.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    t2 = cv.getTickCount()
    time = (t2 - t1) / cv.getTickFrequency()

    cv.rectangle(frame, (10, 2), (250, 20), (255, 255, 255), -1)
    cv.putText(frame, "Blob" + str(" ") + str(time) + str(" Seconds"), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Webcam', frame)

    keyboard = cv.waitKey(30)
    if keyboard == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
