import os
import numpy as np
import cv2

import utils
import data

def generate_clean_annotation(plant):
    annotation_image = utils.load_image(plant.raw_annotation_path)
    mask = ((annotation_image.sum(axis=2) > 70) * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    utils.write_image(opened_mask, plant.annotation_path)

def process_plant(plant):
    # Generate mask
    plant_data = utils.load_json(plant.drone_image_data_path)['labels'][plant.id]
    segment = utils.process_segment(plant_data['segment'])
    drone_image = utils.load_image(plant.drone_image_path)
    mask = cv2.fillPoly(np.zeros(drone_image.shape[:2]), segment, color=255)
    x_min, y_min, x_max, y_max = utils.compute_bounding_box(mask)
    mask_cropped = mask[y_min:y_max, x_min:x_max]
    utils.write_image(mask_cropped, plant.mask_path)
    mask_cropped_bgr = np.repeat(mask_cropped[:, :, np.newaxis], 3, axis=2)
    masked_image_cropped = drone_image[y_min:y_max, x_min:x_max,:] * (mask_cropped_bgr / 255)
    utils.write_image(masked_image_cropped, plant.masked_image_path)

    # Verify that mask x, y, and dimensions match annotation
    annotation_image = utils.load_image(plant.annotation_path)
    assert x_min == plant.x
    assert y_min == plant.y
    assert annotation_image.shape[:2] == mask_cropped.shape

    # Generate tiles
    tile_positions = []
    tile_regions = utils.compute_tile_regions(x_max - x_min, y_max - y_min)
    for tile_id, tile_region in enumerate(tile_regions):
        tile_position, tile_mask, tile, tile_annotation = utils.compute_tile(
            tile_region,
            mask_cropped,
            masked_image_cropped,
            annotation_image,
        )
        tile_positions.append([int(n) for n in tile_position])
        utils.write_image(tile_mask, plant.tile_mask_path(tile_id))
        utils.write_image(tile, plant.tile_path(tile_id))
        utils.write_image(tile_annotation, plant.tile_annotation_path(tile_id))

    # Write info to JSON
    utils.write_json({
        'plant_area': np.count_nonzero(mask),
        'flower_area': np.count_nonzero(annotation_image),
        'bounding_box': [int(n) for n in (x_min, y_min, x_max, y_max)],
        'tile_positions': tile_positions,
    }, plant.data_path)

def main():
    utils.generate_dirs()
    for plant in data.plants:
        generate_clean_annotation(plant)
        process_plant(plant)

if __name__ == '__main__':
   main()
