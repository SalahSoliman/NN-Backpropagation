import numpy as np
import cv2
import argparse
from skimage.exposure import rescale_intensity

def convolve (image, K):
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    pad = (kW-1)//2
    pad = pad

    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y-pad: y+pad+1, x-pad: x+pad+1] #region of interest (ROI)
            k = (roi * K).sum()

            output[y-pad, x-pad] = k
    
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())

smallBlur = np.ones((7, 7), dtype="float") * (1.0/(7.0 * 7.0))
largeBlur = np.ones((21, 21), dtype="float") * (1.0/(21.0 * 21.0))
sharpen = np.array(([0, -1, 0,],[-1, 5, -1], [0, -1, 0]), dtype="int")
laplacian = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]), dtype="int")
sobelX = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), dtype="int")
sobelY = np.array(([-1, -2, 1], [0, 0, 0], [-1, 2, 1]), dtype="int")
emboss = np.array(([-2, -1, 0], [-1, 1, 1], [0, 1, 2]), dtype="int")

kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY),
    ("emboss", emboss)
)

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (kernelName, K) in kernelBank:
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

    cv2.imshow("Original", gray)
    cv2.imshow("{} - convolute fun".format(kernelName), convolveOutput)
    cv2.imshow("{} - openCV".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







