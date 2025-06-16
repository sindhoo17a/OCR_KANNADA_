import keras
import sys
from time import time
import getWord as getWord
from loadingRoutines import loadCNNs
import numpy as np
import cv2 as cv
import os
import fnmatch
import codecs

def output(filepath, text):
    with codecs.open(filepath, "w+", "utf-16") as file:
        file.write(text)

def extract_base_from_filename(filename):
    parts = filename.split('_')
    if len(parts) > 1:
        base_part = parts[-1].replace('.png', '')  # Remove file extension
        try:
            return int(base_part)
        except ValueError:
            print(f"Warning: Unable to parse base from filename '{filename}'")
            return None
    else:
        print(f"Warning: Filename format incorrect for '{filename}'")
        return None

thresh = 192
accMin = 0.3

mainCNN, vattCNN = loadCNNs()
dir = "%s/WordSegmentResult" % sys.path[0]
allFiles = os.listdir(dir)
lines = list()
prefix = "*.png"
someFiles = fnmatch.filter(allFiles, prefix)

for file in someFiles:
    path = "%s/%s" % (dir, file)
    print(f"Processing file: {file}")
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    w, h = img.shape[::-1]
    if h <= 5 or w <= 5:
        continue
    base = extract_base_from_filename(file)
    if base is not None:
        word = getWord.getWord(img, mainCNN, vattCNN, base=None, thresh=thresh, accMin=accMin)
        lines.append(word)

space = '\x20'
text = space.join(lines)

# Ensure the output directory exists
output_dir = "%s/extracted_output" % sys.path[0]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate a unique filename based on the current timestamp
timestamp = int(time())
file = "%s/paragraph_%d.txt" % (output_dir, timestamp)

output(file, text)

print(f"Identification done. Your output is stored in the file \"{file}\"")
