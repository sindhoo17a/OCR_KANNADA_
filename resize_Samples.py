import os
import cv2 as cv

# Define directories
srcdir = 'Unresized/Samples_Gray'
dir1 = 'Samples15'
dir2 = 'Samples25'
dir3 = 'Samples30'
dir4 = 'Samples40'

# Create directories if they don't exist
os.makedirs(dir1, exist_ok=True)
os.makedirs(dir2, exist_ok=True)
os.makedirs(dir3, exist_ok=True)
os.makedirs(dir4, exist_ok=True)

# List all files in the source directory
files = os.listdir(srcdir)

# Process each file
for file in files:
    file_path = os.path.join(srcdir, file)
    if os.path.isfile(file_path):
        img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        if img is not None:
            H, W = img.shape[:2]
            ar = W / H
            W20 = ar * 20
            
            if W20 <= 22:
                img2 = cv.resize(img, (15, 20), interpolation=cv.INTER_CUBIC)
                cv.imwrite(os.path.join(dir1, file), img2)
            elif 18 <= W20 <= 27:
                img2 = cv.resize(img, (25, 20), interpolation=cv.INTER_CUBIC)
                cv.imwrite(os.path.join(dir2, file), img2)
            elif 23 <= W20 <= 37:
                img2 = cv.resize(img, (30, 20), interpolation=cv.INTER_CUBIC)
                cv.imwrite(os.path.join(dir3, file), img2)
            elif W20 >= 33:
                img2 = cv.resize(img, (40, 20), interpolation=cv.INTER_CUBIC)
                cv.imwrite(os.path.join(dir4, file), img2)
