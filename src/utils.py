import os
import re
import json
import shutil
import itertools
import numpy as np
import cv2

import config

def generate_dirs():
    for directory in config.DIRS['processed'].values():
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def load_image(path):
    return cv2.imread(path)

def write_image(image, path):
    written = cv2.imwrite(path, image)
    if not written:
        print(f'Failed to write: {path}')

def process_segment(segment_flat):
    segment_points = np.reshape(segment_flat, (len(segment_flat) // 2, 2))
    return np.int32([segment_points])

def compute_bounding_box(mask):
    # Adapted from https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    cols = np.any(mask, axis=0)
    rows = np.any(mask, axis=1)
    x_min, x_max = np.where(cols)[0][[0, -1]]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    return x_min, y_min, x_max, y_max

def compute_tile_regions(width, height):
    size = config.TILE_SIZE
    overlap = round(size * config.TILE_OVERLAP)
    return [
        (max(x, 0), max(y, 0), min(x + size, width), min(y + size, height))
        for y in itertools.chain(range(0, height - size, size - overlap), [height - size])
        for x in itertools.chain(range(0, width - size, size - overlap), [width - size])
    ]

def compute_tile(region, mask, masked_image, annotation_image):
    size = config.TILE_SIZE
    x_min, y_min, x_max, y_max = region
    width = x_max - x_min
    height = y_max - y_min
    x = round((size - width) / 2)
    y = round((size - height) / 2)
    tile_mask = np.zeros((size, size))
    tile_mask[y:y+height, x:x+width] = mask[y_min:y_max, x_min:x_max]
    tile = np.zeros((size, size, 3))
    tile[y:y+height, x:x+width] = masked_image[y_min:y_max, x_min:x_max]
    tile_annotation = np.zeros((size, size))
    tile_annotation[y:y+height, x:x+width] = annotation_image[y_min:y_max, x_min:x_max]
    return (x_min - x, y_min - y), tile_mask, tile, tile_annotation
