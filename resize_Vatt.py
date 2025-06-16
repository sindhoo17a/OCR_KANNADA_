import os
import cv2 as cv

# Source and target directories
srcdir = 'Unresized/Vattaksharas'
dir1 = 'Vatt15'
dir2 = 'Vatt20'
dir3 = 'Vatt30'

# Create target directories if they do not exist
os.makedirs(dir1, exist_ok=True)
os.makedirs(dir2, exist_ok=True)
os.makedirs(dir3, exist_ok=True)

# List all files in the source directory
valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
files = [f for f in os.listdir(srcdir) if os.path.isfile(os.path.join(srcdir, f)) and os.path.splitext(f)[1].lower() in valid_extensions]

for file in files:
    # Construct full file path
    img_path = os.path.join(srcdir, file)
    
    # Read the image
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded successfully
    if img is None:
        print(f"Warning: Could not read image file {img_path}. Skipping.")
        continue
    
    # Get image dimensions
    H, W = img.shape
    ar = W / H
    W15 = ar * 15
    
    # Resize the image based on width criteria and save to appropriate directory
    if W15 <= 22:
        img2 = cv.resize(img, (15, 15), interpolation=cv.INTER_CUBIC)
        cv.imwrite(os.path.join(dir1, file), img2)
    elif 18 <= W15 <= 27:
        img2 = cv.resize(img, (15, 20), interpolation=cv.INTER_CUBIC)
        cv.imwrite(os.path.join(dir2, file), img2)
    elif W15 >= 23:
        img2 = cv.resize(img, (15, 30), interpolation=cv.INTER_CUBIC)
        cv.imwrite(os.path.join(dir3, file), img2)
