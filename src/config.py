import os

# File paths
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
            'deeplabv2_masks',
            'deeplabv3_masks',
            'tiles',
            'tile_masks',
            'tile_annotations',
            'train',
            'val',
        ]
    }
}
MODEL_DEEPLABV2_PATH = f'{ROOT_DIR}/model/deeplabv2_resnet101_msc-cocostuff164k-100000.pth'
DEEPLABV2_PATH = f'{ROOT_DIR}/model/deeplabv2.pth'
DEEPLABV3_PATH = f'{ROOT_DIR}/model/weights.pt'
XSEDE_V2_DIR = f'{ROOT_DIR}/xsede_deeplabv2'
XSEDE_V3_DIR = f'{ROOT_DIR}/xsede_deeplabv3'

# Deeplab config (from https://github.com/kazuto1011/deeplab-pytorch/blob/master/configs/cocostuff164k.yaml)
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
