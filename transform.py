#imports
import numpy as np
import cv2

def order_points(pts):
    # initialise list of coordinates
    rect = np.zeros((4, 2), dtype = "float32")

    # the top left point will be the smallest sum
    # the bottom right point will be the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    