import numpy as np
import cv2 as cv
import sys
import csv
import os
import fnmatch
from keras.models import load_model

inputShapes = {0: (15, 20), 1: (25, 20), 2: (30, 20), 3: (40, 20)}
vattInputShapes = {0: (15, 15), 1: (20, 15), 2: (30, 15)}

def loadCNNs():
    mainCNN = []
    vattCNN = []
    for i in range(4):
        inputShape = inputShapes[i]
        (img_w, img_h) = inputShape
        file = f"CNNs/CNN{img_w}x{img_h}.h5"
        filepath = os.path.join(os.path.dirname(__file__), file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        model = load_model(filepath)
        print(f"Loaded {file}")
        mainCNN.append(model)
    for i in range(3):
        inputShape = vattInputShapes[i]
        (img_w, img_h) = inputShape
        file = f"CNNs/VattCNN{img_w}x{img_h}.h5"
        filepath = os.path.join(os.path.dirname(__file__), file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        model = load_model(filepath)
        print(f"Loaded {file}")
        vattCNN.append(model)
    return mainCNN, vattCNN

def loadMainTable():
    file = os.path.join(os.path.dirname(__file__), 'LookupTables', 'table.csv')
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        table = {int(row['UID']): (int(row['Char0']), int(row['Char1'])) for row in reader}
    return table

def loadVattTable():
    file = os.path.join(os.path.dirname(__file__), 'LookupTables', 'vatt.csv')
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        table = {int(row['UID']): (int(row['Char0']), int(row['Char1'])) for row in reader}
    return table

def loadMainTrainData(sizeClass):
    widths = {0: 15, 1: 25, 2: 30, 3: 40}
    width = widths[sizeClass]
    dir = os.path.join(os.path.dirname(__file__), 'Samples', 'Resized', f'Samples{width}')
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory not found: {dir}")
    files = os.listdir(dir)
    img_w, img_h = width, 20
    x_train = np.empty((len(files), img_h, img_w), dtype='uint8')
    y_train = np.empty(len(files), dtype='uint16')
    k = 0
    numClasses = 340
    for i in range(numClasses):
        endString = f"*{i:03d}.png"
        classSet = fnmatch.filter(files, endString)
        for file in classSet:
            img = cv.imread(os.path.join(dir, file), cv.IMREAD_GRAYSCALE)
            x_train[k, :, :] = img
            y_train[k] = i
            k += 1
    return x_train, y_train, img_w, img_h

def loadVattTrainData(sizeClass):
    widths = {0: 15, 1: 20, 2: 30}
    width = widths[sizeClass]
    dir = os.path.join(os.path.dirname(__file__), 'Samples', 'Resized', f'Vatt{width}')
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory not found: {dir}")
    files = os.listdir(dir)
    img_w, img_h = width, 15
    x_train = np.empty((len(files), img_h, img_w), dtype='uint8')
    y_train = np.empty(len(files), dtype='uint16')
    k = 0
    numClasses = 32
    for i in range(numClasses):
        endString = f"*{i:03d}.png"
        classSet = fnmatch.filter(files, endString)
        for file in classSet:
            img = cv.imread(os.path.join(dir, file), cv.IMREAD_GRAYSCALE)
            x_train[k, :, :] = img
            y_train[k] = i
            k += 1
    return x_train, y_train, img_w, img_h

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
