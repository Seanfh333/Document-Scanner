from transform import point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

def scan_document(image):
    # compute ratio of the old height to the new height, clone and resize
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    # convert image to greyscale, blur it and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # find contours in the edged image keeping only the largest ones
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    screenCnt = None

    # loop over the contours
    for c in cnts:
        # approximate contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if contour has 4 points assume we have our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None: return None

    # apply the four point transform to get a top down view
    warped = point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # convert the warped to greyscale and threshold it to make it black and white
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    return warped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to the image file")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    warped = scan_document(image)

    if warped is None:
        print("Couldnt find image contours")
    else:
        cv2.imshow("Original", imutils.resize(image, height = 650))
        cv2.imshow("Scanned", imutils.resize(warped, height = 650))
        cv2.waitKey(0)