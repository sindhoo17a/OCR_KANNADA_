from flask import Flask, request, render_template, send_file, redirect, url_for
import os
import sys
import shutil
import numpy as np
import cv2 as cv
import codecs
from time import time
from werkzeug.utils import secure_filename
import getWord as getWord
from loadingRoutines import loadCNNs, extract_base_from_filename

app = Flask(__name__)

app.config['EXTRACTED_FOLDER'] = 'extracted_output'
os.makedirs(app.config['EXTRACTED_FOLDER'], exist_ok=True)

# Function to handle image processing and word segmentation
def process_image(filename):
    import imgProc_util  # Import within the function to avoid circular import issues

    # Parameters
    ysize = 5
    xsize = 13
    thresh = 150

    # Check if the file exists
    if not os.path.exists(filename):
        return f"Error: The file '{filename}' does not exist. Please check the file path.", 400

    # Output directory
    pwd = sys.path[0]
    dirname = 'WordSegmentResult'
    output_dir = os.path.join(pwd, dirname)

    # Clean up previous results
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # Read the image
    A = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    if A is None:
        return f"Error: Could not read the image file '{filename}'. Please check the file path and integrity.", 400

    # Apply threshold
    B = cv.threshold(A, thresh, 255, cv.THRESH_BINARY_INV)[1]
    X = np.ones((ysize, np.shape(B)[1]))
    C = cv.morphologyEx(B, cv.MORPH_CLOSE, X)
    N, D = cv.connectedComponents(C)
    k = 0

    for i in range(1, N):
        (H, W) = np.where(D == i)
        F = B[min(H):max(H) + 1, :]
        ARef = A[min(H):max(H) + 1, :]
        base = imgProc_util.getBase(ARef)
        X2 = np.ones((ysize, xsize))
        C2 = cv.morphologyEx(F, cv.MORPH_DILATE, X2)
        C2 = cv.rotate(C2, cv.ROTATE_90_CLOCKWISE)
        N2, D2 = cv.connectedComponentsWithAlgorithm(C2, 8, cv.CV_16U, ccltype=cv.CCL_WU)
        D2 = cv.rotate(D2, cv.ROTATE_90_COUNTERCLOCKWISE)
        for j in range(1, N2):
            (H2, W2) = np.where(D2 == j)
            F2 = F[:, min(W2):max(W2) + 1]
            ARef2 = ARef[:, min(W2):max(W2) + 1]
            (H2, W2) = np.where(F2 != 0)
            final = ARef2[:, min(W2):max(W2) + 1]
            filepath = os.path.join(output_dir, f"w{str(k).zfill(3)}_{base}.png")
            k += 1
            cv.imwrite(filepath, final)

    return "Segmentation done. Check the 'WordSegmentResult' folder for results."

def combine_words():
    thresh = 192
    accMin = 0.95

    mainCNN, vattCNN = loadCNNs()

    # Process all segmented word images
    word_files = os.listdir('WordSegmentResult')
    word_files.sort()

    lines = []

    for word_file in word_files:
        if word_file.endswith('.png'):
            filepath = os.path.join('WordSegmentResult', word_file)
            img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
            base = extract_base_from_filename(word_file)
            word = getWord.getWord(img, mainCNN, vattCNN, base=None, thresh=thresh, accMin=accMin)
            lines.append(word)

    text = ' '.join(lines)

    # Generate a unique filename based on the current timestamp
    timestamp = int(time())
    output_filepath = os.path.join(app.config['EXTRACTED_FOLDER'], f"paragraph_{timestamp}.txt")

    with codecs.open(output_filepath, "w+", "utf-16") as file:
        file.write(text)

    return output_filepath, text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        filename = os.path.join(sys.path[0], 'uploaded_image.png')
        file.save(filename)
        result = process_image(filename)
        if isinstance(result, tuple):
            return result[0], result[1]  # Error case
        output_filepath, text = combine_words()
        return render_template('result.html', text=text, filename=os.path.basename(output_filepath))

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['EXTRACTED_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)



