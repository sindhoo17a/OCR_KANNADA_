import cv2
import numpy as np
import os

dirname = 'Samples_NEW'
os.makedirs(dirname, exist_ok=True)

fonts = ['Arial', 'Hind', 'Kedage', 'Lohit', 'Nirmala', 'NotoSans', 'NotoSerif', 'Nudi1', 'Nudi5', 'Tunga']
threshes = [192, 128, 128, 128, 128, 128, 128, 128, 128, 128]
prefixes = ['A', 'H', 'K', 'L', 'Ni', 'NSa', 'NSe', 'Nu1_', 'Nu5_', 'T']
lineHeights = [41, 55, 55, 53, 40, 41, 51, 55, 55, 54]
offsets = [230, 233, 230, 230, 230, 228, 228, 229, 229, 232]

print(f"Current working directory: {os.getcwd()}")

for j in range(10):
    font = fonts[j]
    thresh = threshes[j]
    prefix = prefixes[j]
    lineHeight = lineHeights[j]
    offset = offsets[j]

    name = font
    image_path = f'0/{name}.png'
    print(f"Checking if {image_path} exists...")
    
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        continue
    
    A = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if A is None:
        print(f"Warning: Could not open or find the image {image_path}")
        continue
    
    B = A < thresh
    B = B.astype(np.uint8)
    
    X = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) if j != 3 else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    C = cv2.dilate(B, X)
    
    num, D = cv2.connectedComponents(C)
    
    Pts = []
    for i in range(1, num):
        E = (D == i).astype(np.uint8)
        H, W = np.where(E)
        F = B[min(H):max(H), min(W):max(W)]
        HRnd = lineHeight * (np.floor((max(H) - offset) / lineHeight)) + offset
        Pts.append([i, HRnd, min(W)])
    
    Pts = sorted(Pts, key=lambda x: (x[1], x[2]))
    
    for i in range(num - 1):
        E = (D == Pts[i][0]).astype(np.uint8)
        H, W = np.where(E)
        if len(H) == 0 or len(W) == 0:
            print(f"Warning: No valid region found for component {i} in image {name}")
            continue
        
        F = A[min(H):max(H), min(W):max(W)]
        G = F < thresh
        H, W = np.where(G)
        if len(H) == 0 or len(W) == 0:
            print(f"Warning: No valid region found after thresholding for component {i} in image {name}")
            continue
        
        final = F[min(H):max(H), min(W):max(W)]
        
        if final.size == 0:
            print(f"Warning: Empty region after cropping for component {i} in image {name}")
            continue
        
        classes = num // 2
        if i >= classes:
            filepath = os.path.join(dirname, f'{prefix}b{str(i - classes).zfill(3)}.png')
        else:
            filepath = os.path.join(dirname, f'{prefix}{str(i).zfill(3)}.png')
        
        cv2.imwrite(filepath, final)
