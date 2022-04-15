import os
import cv2
from location import Location

# Parameters
DATA_DIR = '../data'
ANNOTATED_DIR = '../annotated'
OUTPUT_DIR = '../processed'
OUTPUT_DIRS = {
    'data': f'{OUTPUT_DIR}/data',
    'mask': f'{OUTPUT_DIR}/masks',
    'masked_image': f'{OUTPUT_DIR}/masked_images',
    'tile': f'{OUTPUT_DIR}/tiles',
    'tile_mask': f'{OUTPUT_DIR}/tile_masks',
    'tile_annotation': f'{OUTPUT_DIR}/tile_annotations',
}
JSON_EXT = 'json'
IMAGE_EXT = 'JPG'
PLANT_TYPE = 'ERFA'
TILE_SIZE = 256
TILE_OVERLAP = 0.25

# Create output directories
for directory in OUTPUT_DIRS.values():
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create location objects
locations = {
    location_name: Location(
        location_name,
        DATA_DIR,
        OUTPUT_DIRS,
        PLANT_TYPE,
        JSON_EXT,
        IMAGE_EXT,
        TILE_SIZE,
        TILE_OVERLAP,
    )
    for location_name in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, location_name))
}

def generate_annotated():
    for file_name in os.listdir(ANNOTATED_DIR):
        name, ext = os.path.splitext(file_name)
        name_components = name.split('_')
        drone_image_name = '_'.join(name_components[:-4])
        plant_id, x, y = (int(n) for n in name_components[-4:-1])
        for location in locations.values():
            if drone_image_name in location.drone_images:
                break
        drone_image = location.drone_images[drone_image_name]
        plant = drone_image.plants[plant_id]
        plant.generate(cv2.imread(f'{ANNOTATED_DIR}/{file_name}'))
        plant_x, plant_y, _, _ = plant.data['bbox']
        if x != plant_x or y != plant_y:
            print(f'x: {x}, y: {y}, plant_x: {plant_x}, plant_y: {plant_y}')

if __name__ == '__main__':
    generate_annotated()


########## OLDER FUNCTIONS ##########

# def add_annotation(image, boundary, bbox, plant_id, color=(0, 0, 255), thickness=8):
#     x_min, y_min, x_max, y_max = bbox
#     pos = ((x_max+x_min)//2, (y_max+y_min)//2)
#     cv2.polylines(image, boundary, isClosed=True, color=color, thickness=thickness)
#     cv2.putText(image, str(plant_id), pos, fontFace=0, fontScale=3, color=color, thickness=thickness)
