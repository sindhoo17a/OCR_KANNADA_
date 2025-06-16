import numpy as np
import cv2 as cv
import sys
import os
import fnmatch

# Get the current working directory
pwd = sys.path[0]
dir = os.path.join(pwd, 'Vattakshara2')
dst = os.path.join(pwd, 'Vattaksharas_CROPPED')

# Check if the directory exists
if not os.path.exists(dir):
    print(f"Error: Directory {dir} does not exist.")
    sys.exit(1)

if not os.path.exists(dst):
    os.makedirs(dst)

allFiles = os.listdir(dir)
prefix = "*03?.png"

for file in allFiles:
    new_img = cv.imread(os.path.join(dir, file), cv.IMREAD_GRAYSCALE)
    retval, ref = cv.threshold(new_img, 128, 255, cv.THRESH_BINARY)
    (H, W) = np.where(ref == 0)
    imgCrop = new_img[min(H):max(H) + 1, min(W):max(W) + 1]
    w, h = imgCrop.shape[::-1]
    dsize = (int(w * 100 / h), 100)
    retval, imgBW = cv.threshold(imgCrop, 128, 255, cv.THRESH_BINARY)
    img64f = cv.Sobel(imgBW, cv.CV_64F, 0, 1, ksize=3)
    img2 = np.uint8(img64f)
    vpp = np.sum(img2, 1) / 255
    pc30 = int(vpp.shape[0] * 3 / 10)
    pc70 = int(vpp.shape[0] * 7 / 10)
    pc80 = int(vpp.shape[0] * 8 / 10)
    if fnmatch.fnmatch(file, prefix):
        base = pc30 + np.argmax(vpp[pc30:pc70])
    else:
        base = pc30 + np.argmax(vpp[pc30:pc80])
    imgLower = imgCrop[base + 1:, :]
    retval, imgLowerBW = cv.threshold(imgLower, 128, 255, cv.THRESH_BINARY)
    pc10 = int(imgLowerBW.shape[0] / 10) + 1
    imgLowerBW[0:pc10] = 255
    (H, W) = np.where(imgLowerBW == 0)
    vatt = imgCrop[base + 1:, min(W):max(W) + 1]
    cv.imwrite(os.path.join(dst, f"c{file}"), vatt)

print("Processing complete.")
