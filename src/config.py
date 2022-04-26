JSON_EXT = 'json'
DRONE_IMAGE_EXT = 'JPG'
OUTPUT_IMAGE_EXT = 'png'
# PLANT_TYPE = 'ERFA'
TILE_SIZE = 256
TILE_OVERLAP = 0.25
DIRS = {
    'data': 'data',
    'annotations': 'annotations',
    'processed': {
        subdir: f'processed/{subdir}'
        for subdir in [
            'info',
            'masks',
            'masked_images',
            'annotations',
            'hsv_masks',
            'tiles',
            'tile_masks',
            'tile_annotations',
        ]
    }
}
