from scan import scan_document
import imutils
import numpy as np
import pytesseract
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
warped = scan_document(image)
config = ("-l eng --oem 1 --psm 6")
text = pytesseract.image_to_string(warped, config=config)
print(text)
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)