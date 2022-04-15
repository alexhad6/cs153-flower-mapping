import os
import re
import json
import random
import itertools
import numpy as np
import cv2

def write_image(image, path):
    if not cv2.imwrite(path, image):
        print(f'Failed to write: {path}')

class Location:
    '''A collection of drone images and their plants.'''

    def __init__(
        self, name,
        data_dir,
        output_dirs,
        plant_type,
        json_ext,
        image_ext,
        tile_size,
        tile_overlap,
    ):
        self.name = name
        self.dir = f'{data_dir}/{name}'
        self.output_dirs = output_dirs
        self.plant_type = plant_type
        self.json_ext = json_ext
        self.image_ext = image_ext
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        # Create drone image objects
        self.drone_images = {
            drone_image_name: DroneImage(self, drone_image_name)
            for drone_image_name, ext in map(os.path.splitext, os.listdir(self.dir))
            if ext == f'.{self.json_ext}'
        }

        # Consolidate all plants into one array
        self.plants = [
            plant
            for drone_image in self.drone_images.values()
            for plant in drone_image.plants.values()
        ]

    def sample_plants(self, num_plants):
        return random.sample(self.plants, num_plants)

    def __repr__(self):
        return f'<Location {self.name}>'

class DroneImage:
    '''A drone image and the plants within it.'''

    def __init__(self, location, name):
        self.location = location
        self.name = name

        # Compute image path
        image_name = re.split(r' \(\d+\)', name)[0]
        self.image_path = f'{location.dir}/{image_name}.{location.image_ext}'

        # Create plant objects based on data
        with open(f'{location.dir}/{name}.{location.json_ext}') as f:
            data = json.load(f)
        self.plants = {
            plant_id: Plant(self, plant_id, plant_data)
            for plant_id, plant_data in enumerate(data['labels'])
            if plant_data['class'] == self.location.plant_type
        }

    @property
    def image(self):
        return cv2.imread(self.image_path)

    def __repr__(self):
        return f'<DroneImage {self.name}>'

class Plant:
    '''A plant within a drone image.'''

    def __init__(self, drone_image, plant_id, data):
        self.drone_image = drone_image
        self.plant_id = plant_id
        self.tile_size = drone_image.location.tile_size
        self.tile_overlap = drone_image.location.tile_overlap

        # Construct plant name and paths
        output_dirs = drone_image.location.output_dirs
        self.name = f'{drone_image.name}_{plant_id}'
        self.mask_path = f'{output_dirs["mask"]}/{self.name}.png'
        self.masked_image_path = f'{output_dirs["masked_image"]}/{self.name}.jpg'
        self.data_path = f'{output_dirs["data"]}/{self.name}.json'
        self.tile_path = f'{output_dirs["tile"]}/{self.name}' + '_{tile_id}.jpg'
        self.tile_mask_path = f'{output_dirs["tile_mask"]}/{self.name}' + '_{tile_id}.png'
        self.tile_annotation_path = f'{output_dirs["tile_annotation"]}/{self.name}' + '_{tile_id}.png'

        # Compute segment polygon
        segment_flat = data['segment']  # 1D array of [x0, y0, x1, y1, ...]
        segment_points = np.reshape(segment_flat, (len(segment_flat) // 2, 2))
        self.segment = np.int32([segment_points])

    @property
    def data(self):
        assert os.path.isfile(self.data_path), f'No data for {self.name}; call generate first'
        with open(f'{self.data_path}') as f:
            return json.load(f)

    @property
    def mask(self):
        assert os.path.isfile(self.mask_path), f'No mask for {self.name}; call generate first'
        return cv2.imread(self.mask_path)

    @property
    def masked_image(self):
        assert os.path.isfile(self.masked_image_path), f'No masked image for {self.name}; call generate first'
        return cv2.imread(self.masked_image_path)

    def write_data(self, data):
        with open(f'{self.data_path}', 'w') as f:
            json.dump(data, f)

    def compute_bounding_box(self, mask):
        # Adapted from https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
        cols = np.any(mask, axis=0)
        rows = np.any(mask, axis=1)
        x_min, x_max = np.where(cols)[0][[0, -1]]
        y_min, y_max = np.where(rows)[0][[0, -1]]
        return x_min, y_min, x_max, y_max

    def compute_tile_regions(self, width, height):
        size = self.tile_size
        overlap = round(size * self.tile_overlap)
        return [
            (max(x, 0), max(y, 0), min(x + size, width), min(y + size, height))
            for y in itertools.chain(range(0, height - size, size - overlap), [height - size])
            for x in itertools.chain(range(0, width - size, size - overlap), [width - size])
        ]

    def compute_tile(self, region, mask, masked_image, annotation_image=None):
        x_min, y_min, x_max, y_max = region
        width = x_max - x_min
        height = y_max - y_min
        x = round((self.tile_size - width) / 2)
        y = round((self.tile_size - height) / 2)
        tile_mask = np.zeros((self.tile_size, self.tile_size))
        tile_mask[y:y+height, x:x+width] = mask[y_min:y_max, x_min:x_max]
        tile = np.zeros((self.tile_size, self.tile_size, 3))
        tile[y:y+height, x:x+width] = masked_image[y_min:y_max, x_min:x_max]
        if annotation_image is None:
            tile_annotation = None
        else:
            tile_annotation = np.zeros((self.tile_size, self.tile_size, 3))
            tile_annotation[y:y+height, x:x+width] = annotation_image[y_min:y_max, x_min:x_max]
        return (x_min - x, y_min - y), tile_mask, tile, tile_annotation

    def generate(self, annotation_image=None):
        # Load in the image
        image = self.drone_image.image

        # Generate mask and bounding box
        mask = cv2.fillPoly(np.zeros(image.shape[:2]), self.segment, color=255)
        x_min, y_min, x_max, y_max = self.compute_bounding_box(mask)
        mask_cropped = mask[y_min:y_max, x_min:x_max]
        write_image(mask_cropped, self.mask_path)

        # Mask and crop the image
        mask_cropped_bgr = np.repeat(mask_cropped[:, :, np.newaxis], 3, axis=2)
        masked_image_cropped = image[y_min:y_max, x_min:x_max,:] * (mask_cropped_bgr / 255)
        write_image(masked_image_cropped, self.masked_image_path)

        # Compute tiles
        tile_positions = []
        tile_regions = self.compute_tile_regions(x_max - x_min, y_max - y_min)
        for tile_id, tile_region in enumerate(tile_regions):
            tile_position, tile_mask, tile, tile_annotation = self.compute_tile(tile_region, mask_cropped, masked_image_cropped, annotation_image)
            tile_positions.append([int(n) for n in tile_position])
            write_image(tile_mask, self.tile_mask_path.format(tile_id=tile_id))
            write_image(tile, self.tile_path.format(tile_id=tile_id))
            if tile_annotation is not None:
                write_image(tile_annotation, self.tile_annotation_path.format(tile_id=tile_id))

        # Compute area and write data to JSON
        self.write_data({
            'area': np.count_nonzero(mask),
            'bbox': [int(n) for n in (x_min, y_min, x_max, y_max)],
            'tile_positions': tile_positions,
        })

    def __repr__(self):
        return f'<Plant {self.name}>'
