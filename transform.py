#imports
import numpy as np
import cv2

def point_order(pts):
    # initialise list of coordinates
    coords = np.zeros((4, 2), dtype = "float32")

    # the top left point will be the smallest sum
    # the bottom right point will be the largest sum
    # coordinates are assigned in clockwise order so that
    # 0 = top left (smallest sum)
    # 1 = top right (smallest diff)
    # 2 = bottom right (biggest sum)
    # 3 = bottom left (biggest diff)
    s = pts.sum(axis = 1)
    coords[0] = pts[np.argmin(s)]
    coords[2] = pts[np.argmax(s)]

    #difference between the points to find bottom left and top right
    differnce = np.diff(pts, axis = 1)
    coords[1] = pts[np.argmin(differnce)]
    coords[3] = pts[np.argmax(differnce)]

    return coords


def point_transform(image, pts):
    #order the points
    coords = point_order(pts)
    #assign each coordinate its own variable
    (tl, tr, br, bl) = coords

    # compute the width of new image (distance between bottom/top left and right x coords)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of neew image (max distance between top and bottom of l and r)
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(coords, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped