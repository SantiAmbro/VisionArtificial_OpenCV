import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='This program shows different types of contours hierarchy')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='RETR_LIST')
args = parser.parse_args()

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    resizedFrame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
    # gray = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(gray, 127, 255, 0)
    blurred_frame = cv2.GaussianBlur(resizedFrame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([38, 86, 0])
    upper_blue = np.array([121, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    if args.algo == 'RETR_LIST':
        contours, h = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print('Retr_List Hierarchy: \n' + str(h))
        for contour in contours:
            cv2.drawContours(resizedFrame, contour, -1, (0, 255, 0), 3)
    elif args.algo == 'RETR_EXTERNAL':
        contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print('Retr_External Hierarchy: \n' + str(h))
        for contour in contours:
            cv2.drawContours(resizedFrame, contour, -1, (0, 255, 0), 3)
    elif args.algo == 'RETR_CCOMP':
        contours, h = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        print('Retr_CComp Hierarchy: \n' + str(h))
        for contour in contours:
            cv2.drawContours(resizedFrame, contour, -1, (0, 255, 0), 3)
    elif args.algo == 'RETR_TREE':
        contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print('Retr_Tree Hierarchy: \n' + str(h))
        for contour in contours:
            cv2.drawContours(resizedFrame, contour, -1, (0, 255, 0), 3)

    cv2.imshow('Frame ' + args.algo + ' Contour', resizedFrame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1)
    if key == ord('a'):
        args.algo = 'RETR_LIST'
    elif key == ord('s'):
        args.algo = 'RETR_EXTERNAL'
    elif key == ord('d'):
        args.algo = 'RETR_CCOMP'
    elif key == ord('f'):
        args.algo = 'RETR_TREE'
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
