import os

JSON_EXT = 'json'
DRONE_IMAGE_EXT = 'JPG'
OUTPUT_IMAGE_EXT = 'png'
TILE_SIZE = 256
TILE_OVERLAP = 0.25
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DIRS = {
    'data': os.path.join(ROOT_DIR, 'data'),
    'annotations': os.path.join(ROOT_DIR, 'annotations'),
    'processed': {
        subdir: os.path.join(ROOT_DIR, 'processed', subdir)
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

MODEL_PATH = f'{ROOT_DIR}/model/deeplabv2_resnet101_msc-cocostuff164k-100000.pth'
COCOSTUFF_LABELS_PATH = f'{ROOT_DIR}/src/deeplab/labels.txt'

# Download test images and put their paths here
TEST_IMAGE_PATHS = {
    'grass': f'{ROOT_DIR}/model/grass.jpg',
    'forest': f'{ROOT_DIR}/model/forest.jpg',
    'flowers': f'{ROOT_DIR}/model/flowers.jpg',
    'apple-tree': f'{ROOT_DIR}/model/apple-tree.jpg',
    'birds': f'{ROOT_DIR}/model/birds.jpg',
    'tile1': f'{ROOT_DIR}/model/tile1.png',
    'tile2': f'{ROOT_DIR}/model/tile2.png',
    'tile3': f'{ROOT_DIR}/model/tile3.png',
}

# Deeplab config
# From https://github.com/kazuto1011/deeplab-pytorch/blob/master/configs/cocostuff164k.yaml
COCOSTUFF_N_CLASSES = 182
COCOSTUFF_MEAN = {
    'r': 122.675,
    'g': 116.669,
    'b': 104.008,
}
CRF = {
    'iter_max': 10,
    'pos_w': 3,
    'pos_xy_std': 1,
    'bi_w': 4,
    'bi_xy_std': 67,
    'bi_rgb_std': 3,
}
