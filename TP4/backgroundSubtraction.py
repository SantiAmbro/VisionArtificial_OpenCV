from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

def nothing(x):
    pass


def originMethod():
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                  OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    args = parser.parse_args()

    cv.namedWindow("FG Mask")
    if args.algo == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2(detectShadows=True)
        cv.createTrackbar("Learning Rate:", "FG Mask", 0, 1, nothing)
        cv.createTrackbar("Shadows:", "FG Mask", 0, 255, nothing)
    else:
        backSub = cv.createBackgroundSubtractorKNN(detectShadows=True)
        cv.createTrackbar("Learning Rate:", "FG Mask", 0, 1, nothing)
        cv.createTrackbar("Shadows:", "FG Mask", 0, 255, nothing)

    capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))

    if not capture.isOpened:
        print('Unable to open: ' + args.input)
        exit(0)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        lr = cv.getTrackbarPos("Learning Rate:", "FG Mask")

        s = cv.getTrackbarPos("Shadows:", "FG Mask")
        backSub.setShadowValue(s)

        fgMask = backSub.apply(frame, lr)

        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.rectangle(fgMask, (10, 2), (110, 20), (255, 255, 255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.putText(fgMask, str(args.algo) + str(" ") + str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fgMask)

        keyboard = cv.waitKey(30)
        if keyboard == ord('m'):
            args.algo = 'MOG2'
        elif keyboard == ord('k'):
            args.algo = 'KNN'
        elif keyboard == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()


def specialMethod():
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                      OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    args = parser.parse_args()

    # Subtractors
    mogSubtractor = cv.bgsegm.createBackgroundSubtractorMOG(300)
    mog2Subtractor = cv.createBackgroundSubtractorMOG2(300, 400, True)
    gmgSubtractor = cv.bgsegm.createBackgroundSubtractorGMG(10, .8)
    knnSubtractor = cv.createBackgroundSubtractorKNN(100, 400, True)
    cntSubtractor = cv.bgsegm.createBackgroundSubtractorCNT(5, True)
    gsocSubtractor = cv.bgsegm.createBackgroundSubtractorGSOC()
    lsbpSubtractor = cv.bgsegm.createBackgroundSubtractorLSBP()

    capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))

    if not capture.isOpened:
        print('Unable to open: ' + args.input)
        exit(0)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        resizedFrame = cv.resize(frame, (0, 0), fx=0.4, fy=0.4)
        mogMask = mogSubtractor.apply(resizedFrame)
        mog2Mask = mog2Subtractor.apply(resizedFrame)
        gmgMask = gmgSubtractor.apply(resizedFrame)
        gmgMask = cv.morphologyEx(gmgMask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        knnMask = knnSubtractor.apply(resizedFrame)
        cntMask = cntSubtractor.apply(resizedFrame)
        gsocMask = gsocSubtractor.apply(resizedFrame)
        lsbpMask = lsbpSubtractor.apply(resizedFrame)
        cv.rectangle(resizedFrame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.rectangle(mogMask, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.rectangle(mog2Mask, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.rectangle(gmgMask, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.rectangle(knnMask, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.rectangle(cntMask, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.rectangle(gsocMask, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.rectangle(lsbpMask, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(resizedFrame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv.putText(mogMask, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv.putText(mog2Mask, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv.putText(gmgMask, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv.putText(knnMask, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv.putText(cntMask, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv.putText(gsocMask, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv.putText(lsbpMask, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('Frame', resizedFrame)
        cv.imshow('MOG', mogMask)
        cv.imshow('MOG2', mog2Mask)
        cv.imshow('GMG', gmgMask)
        cv.imshow('KNN', knnMask)
        cv.imshow('CNT', cntMask)
        cv.imshow('GSOC', gsocMask)
        cv.imshow('LSBP', lsbpMask)
        cv.moveWindow('Frame', 0, 0)
        cv.moveWindow('MOG2', 0, 325)
        cv.moveWindow('KNN', 0, 625)
        cv.moveWindow('GMG', 500, 0)
        cv.moveWindow('MOG', 500, 325)
        cv.moveWindow('CNT', 500, 625)
        cv.moveWindow('GSOC', 1000, 0)
        cv.moveWindow('LSBP', 1000, 325)

        keyboard = cv.waitKey(30)
        if keyboard == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()

specialMethod()
