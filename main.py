# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, send_file, redirect, url_for
import os
import sys
import shutil
import numpy as np
import cv2 as cv
import codecs
from time import time
from werkzeug.utils import secure_filename
import getWord
from loadingRoutines import loadCNNs, extract_base_from_filename
import pytesseract
from PIL import Image
import logging
import re
import platform

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# --- Tesseract Configuration for Windows ---

WINDOWS_TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # EXAMPLE PATH

if platform.system() == "Windows":
    logging.info("Windows OS detected. Checking Tesseract path...")
    if os.path.exists(WINDOWS_TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT_PATH
        logging.info(f"Tesseract path set to: {WINDOWS_TESSERACT_PATH}")
    else:
        logging.warning(f"Tesseract executable NOT FOUND at specified path: {WINDOWS_TESSERACT_PATH}")
        logging.warning("Fallback OCR may fail. Install Tesseract or correct the path in the script.")

app.config['EXTRACTED_FOLDER'] = 'extracted_output'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SEGMENT_FOLDER'] = 'WordSegmentResult'

os.makedirs(app.config['EXTRACTED_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def process_image(filename):
    try:
        import imgProc_util
    except ImportError:
        logging.error("Error: Could not import 'imgProc_util'. Ensure it is installed/accessible.")
        return None, "Error: Missing required image processing utility module.", 500

    ysize = 5
    xsize = 13
    thresh = 150
    output_dir = os.path.join(os.getcwd(), app.config['SEGMENT_FOLDER'])

    if not os.path.exists(filename):
        logging.error(f"Error: Input file '{filename}' does not exist.")
        return None, f"Error: Input file '{filename}' does not exist.", 400

    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except OSError as e:
            logging.error(f"Error removing directory {output_dir}: {e}")
            return None, f"Error cleaning up segmentation directory.", 500
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Error creating directory {output_dir}: {e}")
        return None, f"Error creating segmentation directory.", 500

    try:
        A = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        if A is None:
            if not os.access(filename, os.R_OK):
                 logging.error(f"Error reading image file '{filename}': Permission denied or file not accessible.")
                 return None, f"Error: Cannot access image file '{filename}'. Check permissions.", 400
            raise ValueError("Image data is None after reading. File might be corrupted or not a supported image format.")
    except Exception as e:
        logging.error(f"Error reading image file '{filename}': {e}")
        return None, f"Error: Could not read image file '{filename}'. It might be corrupted or an unsupported format.", 400

    if A.size == 0:
        logging.error(f"Image '{filename}' is empty or could not be decoded.")
        return None, "Error: Image appears to be empty or is not a valid image file.", 400

    try:
        B = cv.threshold(A, thresh, 255, cv.THRESH_BINARY_INV)[1]
    except cv.error as e:
        logging.error(f"OpenCV error during thresholding: {e}")
        return None, "Error: Failed during image thresholding.", 500
    except Exception as e:
        logging.error(f"Unexpected error during thresholding: {e}")
        return None, "Error: Unexpected issue during image thresholding.", 500

    X = np.ones((ysize, np.shape(B)[1]))
    try:
        C = cv.morphologyEx(B, cv.MORPH_CLOSE, X)
        N, D = cv.connectedComponents(C)
    except cv.error as e:
        logging.error(f"OpenCV error during line segmentation (morphology/connectedComponents): {e}")
        return None, "Error: Failed during line segmentation.", 500
    except Exception as e:
        logging.error(f"Unexpected error during line segmentation: {e}")
        return None, "Error: Unexpected issue during line segmentation.", 500

    lines_data = []
    global_word_counter = 0

    line_bounding_boxes = []
    for i in range(1, N):
        (H, W) = np.where(D == i)
        if H.size == 0 or W.size == 0: continue
        min_h, max_h = min(H), max(H)
        min_w, max_w = min(W), max(W)
        line_bounding_boxes.append(((min_h, max_h, min_w, max_w), i))

    line_bounding_boxes.sort(key=lambda item: item[0][0])

    for line_box_info in line_bounding_boxes:
        (min_h, max_h, min_w, max_w), label_i = line_box_info

        F_line = B[min_h:max_h + 1, min_w:max_w + 1]
        ARef_line = A[min_h:max_h + 1, min_w:max_w + 1]
        if F_line.size == 0 or ARef_line.size == 0: continue

        try:
            base = imgProc_util.getBase(ARef_line)
            if not isinstance(base, str):
                 logging.warning(f"imgProc_util.getBase returned non-string type for line. Using default 'unknown'.")
                 base = "unknown"
            base = re.sub(r'[\\/*?:"<>|]', '_', base)
        except Exception as e:
            logging.warning(f"Could not get base for line, using default 'unknown'. Error: {e}")
            base = "unknown"

        X2 = np.ones((ysize, xsize))
        try:
            C2_line = cv.morphologyEx(F_line, cv.MORPH_DILATE, X2)
            N2, D2_line = cv.connectedComponents(C2_line)
        except cv.error as e:
            logging.warning(f"OpenCV error during word segmentation for a line: {e}. Skipping line.")
            continue
        except Exception as e:
            logging.warning(f"Unexpected error during word segmentation for a line: {e}. Skipping line.")
            continue

        words_in_line = []

        word_components = []
        for j in range(1, N2):
            (H2, W2) = np.where(D2_line == j)
            if H2.size == 0 or W2.size == 0: continue
            min_h2, max_h2 = min(H2), max(H2)
            min_w2, max_w2 = min(W2), max(W2)
            word_components.append(((min_h2, max_h2, min_w2, max_w2), j))

        word_components.sort(key=lambda item: item[0][2])

        for word_box_info in word_components:
             (min_h_word_rel, max_h_word_rel, min_w_word_rel, max_w_word_rel), label_j = word_box_info

             word_mask = (D2_line == label_j).astype(np.uint8)

             h_mask, w_mask = np.where(word_mask > 0)
             if h_mask.size == 0 or w_mask.size == 0: continue
             min_h_mask, max_h_mask = min(h_mask), max(h_mask)
             min_w_mask, max_w_mask = min(w_mask), max(w_mask)

             final_word = ARef_line[min_h_mask:max_h_mask + 1, min_w_mask:max_w_mask + 1]

             if final_word.size == 0: continue

             padding = 2
             try:
                 final_word_padded = cv.copyMakeBorder(final_word, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=[255])
             except cv.error as e:
                 logging.warning(f"OpenCV error adding padding to word image: {e}. Skipping word.")
                 continue

             word_filename = f"l{len(lines_data):02d}_w{str(global_word_counter).zfill(4)}_{base}.png"
             filepath = os.path.join(output_dir, word_filename)

             try:
                 is_success, im_buf_arr = cv.imencode(".png", final_word_padded)
                 if is_success:
                     im_buf_arr.tofile(filepath)
                     x_coordinate_global = min_w + min_w_mask
                     words_in_line.append((filepath, x_coordinate_global))
                     global_word_counter += 1
                 else:
                      logging.warning(f"Could not encode word image buffer for {filepath}. Skipping word.")

             except IOError as e:
                 logging.warning(f"Could not write word image file {filepath}. Error: {e}")
             except Exception as e:
                 logging.warning(f"Unexpected error writing word image {filepath}. Error: {e}")

        if words_in_line:
            words_in_line.sort(key=lambda item: item[1])
            lines_data.append([item[0] for item in words_in_line])

    if not lines_data:
        logging.warning(f"No words were successfully segmented from image '{filename}'.")
        return [], "Segmentation completed, but no words were isolated.", 200
    else:
        logging.info(f"Segmentation done. {global_word_counter} words saved, structured in {len(lines_data)} lines.")
        return lines_data, "Segmentation successful.", 200

def perform_primary_ocr(lines_structure):
    logging.info("Attempting primary OCR using custom CNN models (preserving lines)...")
    thresh = 192
    accMin = 0.95

    if not lines_structure:
        logging.warning("Received empty structure for primary OCR. Skipping.")
        return None, ""

    try:
        mainCNN, vattCNN = loadCNNs()
        if mainCNN is None or vattCNN is None:
             raise ValueError("One or both CNN models failed to load.")
    except ImportError:
         logging.error("Failed to import loadingRoutines. Ensure the module is available.")
         return None, ""
    except Exception as e:
        logging.error(f"Failed to load CNN models via loadingRoutines: {e}")
        return None, ""

    processed_lines = []
    total_word_count = 0

    for line_index, word_files_in_line in enumerate(lines_structure):
        recognized_words_in_line = []
        for word_index, filepath in enumerate(word_files_in_line):
            if not os.path.exists(filepath):
                logging.warning(f"Word file missing during primary OCR: {filepath}. Skipping.")
                continue
            try:
                img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
                if img is None:
                     try:
                         pil_img = Image.open(filepath).convert('L')
                         img = np.array(pil_img)
                         logging.info(f"Successfully read {filepath} using PIL fallback.")
                     except Exception as pil_e:
                         logging.warning(f"Could not read word image with OpenCV or PIL: {filepath}. Error: {pil_e}. Skipping.")
                         continue
                elif img.size == 0:
                    logging.warning(f"Read empty word image (size 0): {filepath}. Skipping.")
                    continue

                # base = extract_base_from_filename(os.path.basename(filepath)) # Uncomment if needed by getWord
                word = getWord.getWord(img, mainCNN, vattCNN, base=None, thresh=thresh, accMin=accMin)

                if word and isinstance(word, str) and word.strip():
                    recognized_words_in_line.append(word.strip())
                    total_word_count += 1

            except ImportError:
                 logging.error("Failed to import getWord. Ensure the module is available. Aborting primary OCR.")
                 return None, ""
            except Exception as e:
                logging.warning(f"Error processing word file {filepath} with primary OCR: {e}. Skipping word.")
                continue

        if recognized_words_in_line:
            line_text = ' '.join(recognized_words_in_line)
            processed_lines.append(line_text)

    if not processed_lines:
        logging.warning("Primary OCR completed, but no text was recognized in any line.")
        return None, ""

    final_text = '\n'.join(processed_lines)
    logging.info(f"Primary OCR successful. Recognized {total_word_count} words across {len(processed_lines)} lines.")

    timestamp = int(time())
    output_filename = f"paragraph_primary_{timestamp}.txt"
    output_filepath = os.path.join(app.config['EXTRACTED_FOLDER'], output_filename)

    try:
        with codecs.open(output_filepath, "w+", "utf-16") as file:
            file.write(final_text)
        logging.info(f"Primary OCR result saved to {output_filepath}")
        return output_filepath, final_text
    except IOError as e:
        logging.error(f"Failed to write primary OCR output to {output_filepath}: {e}")
        return None, final_text
    except Exception as e:
        logging.error(f"Unexpected error writing primary OCR output to {output_filepath}: {e}")
        return None, final_text

def perform_fallback_ocr(image_path):
    logging.info(f"Attempting fallback OCR using Tesseract on {image_path}...")
    if not os.path.exists(image_path):
         logging.error(f"Fallback OCR failed: Image file not found at {image_path}")
         return ""
    try:
        img = Image.open(image_path)
        custom_config = r'--oem 1 --psm 3 -l kan'
        text = pytesseract.image_to_string(img, config=custom_config)
        text = text.strip()
        if text:
            logging.info("Fallback OCR successful using Tesseract.")
        else:
            logging.warning("Fallback OCR with Tesseract completed but returned empty result.")
        return text
    except pytesseract.TesseractNotFoundError:
        logging.error("Fallback OCR failed: Tesseract is not installed or not configured correctly.")
        logging.error("Ensure Tesseract is installed and path set correctly (Windows) or in PATH (Linux/macOS).")
        return ""
    except FileNotFoundError:
         logging.error(f"Fallback OCR failed: Could not find or open image file at {image_path} using PIL.")
         return ""
    except Exception as e:
        logging.error(f"An error occurred during fallback Tesseract OCR on {image_path}: {e}")
        return ""

def count_words(text):
    if not text or not isinstance(text, str):
        return 0
    words = re.split(r'\s+', text.strip())
    words = [word for word in words if word]
    return len(words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['EXTRACTED_FOLDER'], filename)
    if os.path.exists(file_path):
        try:
            try:
                safe_filename = filename.encode('utf-8').decode('latin-1')
            except UnicodeDecodeError:
                 safe_filename = 'downloaded_text.txt'
            return send_file(file_path, as_attachment=True, download_name=safe_filename)
        except Exception as e:
             logging.error(f"Error sending file {filename}: {e}")
             return "Error sending file.", 500
    else:
        logging.warning(f"Download request for non-existent file: {filename}")
        return "File not found.", 404

@app.route('/upload', methods=['POST'])
def upload_file():
    WORD_COUNT_THRESHOLD_FACTOR = 1.1

    if 'file' not in request.files:
        logging.warning("Upload attempt with no 'file' part in the request.")
        return "No file part in the request.", 400
    file = request.files['file']
    if file.filename == '':
        logging.warning("Upload attempt with no selected file.")
        return "No selected file.", 400

    if file:
        original_filename = secure_filename(file.filename)
        base, ext = os.path.splitext(original_filename)
        timestamp = int(time())
        base = re.sub(r'[\\/*?:"<>|]', '_', base)
        unique_filename = f"{base}_{timestamp}{ext}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            file.save(image_path)
            logging.info(f"File uploaded and saved to {image_path}")
        except Exception as e:
            logging.error(f"Failed to save uploaded file to {image_path}: {e}")
            return "Error saving uploaded file.", 500

        lines_structure, seg_message, seg_status_code = process_image(image_path)

        if lines_structure is None:
             logging.error(f"Segmentation failed for {image_path}. Message: {seg_message}")
             return seg_message, seg_status_code

        if not lines_structure:
             logging.warning(f"Segmentation found no words in {image_path}. Proceeding directly to fallback OCR.")
             pass

        primary_filepath, primary_text = None, ""
        if lines_structure:
             primary_filepath, primary_text = perform_primary_ocr(lines_structure)
             primary_text = primary_text or ""

        fallback_text = perform_fallback_ocr(image_path)
        fallback_text = fallback_text or ""

        primary_word_count = count_words(primary_text)
        fallback_word_count = count_words(fallback_text)
        logging.info(f"OCR Word Counts - Primary: {primary_word_count}, Fallback (Tesseract): {fallback_word_count}")

        final_text = ""
        final_filepath = None
        chosen_method = "None"

        if primary_word_count > 0 and fallback_word_count > 0:
            if primary_word_count >= fallback_word_count:
                logging.info(f"Choosing Primary OCR result (word count {primary_word_count} >= {fallback_word_count}).")
                final_text = primary_text
                final_filepath = primary_filepath
                chosen_method = "primary"
            elif fallback_word_count > primary_word_count:
                 logging.info(f"Choosing Fallback OCR result (word count {fallback_word_count} > {primary_word_count}).")
                 final_text = fallback_text
                 chosen_method = "fallback"

        elif primary_word_count > 0:
            logging.info("Choosing Primary OCR result (fallback produced no words).")
            final_text = primary_text
            final_filepath = primary_filepath
            chosen_method = "primary"

        elif fallback_word_count > 0:
            logging.info("Choosing Fallback OCR result (primary produced no words or was skipped).")
            final_text = fallback_text
            chosen_method = "fallback"

        else:
            logging.error("Both primary and fallback OCR failed to produce any text.")
            final_text = "[OCR Failed: Could not extract text from image]"
            chosen_method = "none"
            final_filepath = None

        if chosen_method == "fallback" and final_text and final_filepath is None:
            ts = int(time())
            fallback_filename = f"paragraph_fallback_{ts}.txt"
            fallback_filepath = os.path.join(app.config['EXTRACTED_FOLDER'], fallback_filename)
            try:
                with codecs.open(fallback_filepath, "w+", "utf-16") as fb_file:
                    fb_file.write(final_text)
                logging.info(f"Fallback OCR result saved to {fallback_filepath}")
                final_filepath = fallback_filepath
            except IOError as e:
                logging.error(f"Failed to write chosen fallback OCR output to {fallback_filepath}: {e}")
                final_filepath = None
            except Exception as e:
                 logging.error(f"Unexpected error writing fallback OCR output to {fallback_filepath}: {e}")
                 final_filepath = None

        segment_folder = os.path.join(os.getcwd(), app.config['SEGMENT_FOLDER'])
        if os.path.exists(segment_folder):
             try:
                 shutil.rmtree(segment_folder)
                 logging.info(f"Cleaned up segmentation folder: {segment_folder}")
             except Exception as e:
                 logging.warning(f"Could not clean up segmentation folder {segment_folder}: {e}")

        result_filename = os.path.basename(final_filepath) if final_filepath and os.path.exists(final_filepath) else None

        return render_template('result.html',
                               text=final_text,
                               filename=result_filename,
                               chosen_method=chosen_method,
                               primary_wc=primary_word_count,
                               fallback_wc=fallback_word_count
                              )

    logging.error("File upload handling failed unexpectedly after file check.")
    return "An unexpected error occurred during file upload processing.", 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)




