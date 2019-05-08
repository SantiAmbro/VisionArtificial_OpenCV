import cv2 as cv
import argparse


def nothing(x):
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--facedet', type=str, help='Feature Detection method (FAST, AGAST, AKAZE, ORB).', default='FAST')
args = parser.parse_args()

cv.namedWindow("Webcam")

capture = cv.VideoCapture(0)
fast = cv.FastFeatureDetector_create()
agast = cv.AgastFeatureDetector_create()
akaze = cv.AKAZE_create(threshold=0.001)
orb = cv.ORB_create(nfeatures=1500)

if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
time = 0
while True:
    ret, frame = capture.read()
    if frame is None:
        break

    if args.facedet == 'FAST':
        t1 = cv.getTickCount()
        th = cv.getTrackbarPos("Threshold: ", "Webcam")
        cv.createTrackbar("Threshold: ", "Webcam", 10, 10000, nothing)
        fast.setThreshold(th)
        keypoints = fast.detect(frame, None)
        frame = cv.drawKeypoints(frame, keypoints, None)
        t2 = cv.getTickCount()
        time = (t2 - t1)/cv.getTickFrequency()
    elif args.facedet == 'AGAST':
        t1 = cv.getTickCount()
        th = cv.getTrackbarPos("Threshold: ", "Webcam")
        agast.setThreshold(th)
        keypoints = agast.detect(frame, None)
        frame = cv.drawKeypoints(frame, keypoints, None)
        t2 = cv.getTickCount()
        time = (t2 - t1) / cv.getTickFrequency()
    elif args.facedet == 'AKAZE':
        t1 = cv.getTickCount()
        th = cv.getTrackbarPos("Threshold: ", "Webcam")
        keypoints = akaze.detect(frame, None)
        frame = cv.drawKeypoints(frame, keypoints, None)
        t2 = cv.getTickCount()
        time = (t2 - t1) / cv.getTickFrequency()
    elif args.facedet == 'ORB':
        t1 = cv.getTickCount()
        keypoints, descriptors = orb.detectAndCompute(frame, None)
        frame = cv.drawKeypoints(frame, keypoints, None)
        t2 = cv.getTickCount()
        time = (t2 - t1) / cv.getTickFrequency()

    cv.rectangle(frame, (10, 2), (250, 20), (255, 255, 255), -1)
    cv.putText(frame, str(args.facedet) + str(" ") + str(time) + str(" Seconds"), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Webcam', frame)

    keyboard = cv.waitKey(30)
    if keyboard == ord('f'):
        args.facedet = 'FAST'
    elif keyboard == ord('a'):
        args.facedet = 'AGAST'
    elif keyboard == ord('k'):
        args.facedet = 'AKAZE'
    elif keyboard == ord('o'):
        args.facedet = 'ORB'
    elif keyboard == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
