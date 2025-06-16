import cv2
import numpy as np
import os

# Create directory
dirname = 'Special'
os.makedirs(dirname, exist_ok=True)

# Threshold value
thresh = 192

# Load the image
image_path = '0/Special.png'
A = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if A is None:
    raise FileNotFoundError(f"Could not open or find the image {image_path}")

# Thresholding
B = A < thresh

# Dilation with a line structuring element
X = np.ones((1, B.shape[1]), dtype=np.uint8)
C = cv2.dilate(B.astype(np.uint8), X)

# Label connected components
num, D = cv2.connectedComponents(C)

k = 0

for i in range(1, num):
    E = (D == i).astype(np.uint8)
    H, W = np.where(E)
    F = B[min(H):max(H)+1, min(W):max(W)+1]
    AgRef = A[min(H):max(H)+1, min(W):max(W)+1]

    # Further dilation with a disk structuring element
    X2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    C2 = cv2.dilate(F.astype(np.uint8), X2)

    # Label connected components in the dilated image
    num2, D2 = cv2.connectedComponents(C2)

    for j in range(1, num2):
        E2 = (D2 == j).astype(np.uint8)
        H2, W2 = np.where(E2)
        F2 = F[min(H2):max(H2)+1, min(W2):max(W2)+1]
        AgRef2 = AgRef[min(H2):max(H2)+1, min(W2):max(W2)+1]
        H2, W2 = np.where(F2)
        final = AgRef2[min(H2):max(H2)+1, min(W2):max(W2)+1]

        filepath = os.path.join(dirname, f'Spcl_{k//18:02d}{322+k%18:03d}.png')
        k += 1

        cv2.imwrite(filepath, final)

print("Processing complete.")
