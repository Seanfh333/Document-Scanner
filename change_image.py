from transform import point_transform
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-c", "--coords", help = "comma seperated list of source points")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype = "float32")