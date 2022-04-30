import os
import shutil
import random

# Local imports
import config

def generate_train_val():
    tiles = os.listdir(config.DIRS['processed']['tiles'])
    random.shuffle(tiles)
    train_num = round(len(tiles) * 0.7)
    tiles = {
        'train': tiles[:train_num],
        'val': tiles[train_num:],
    }
    for stage in ['train', 'val']:
        for tile in tiles[stage]:
            shutil.copyfile(
                src=os.path.join(config.DIRS['processed']['tiles'], tile),
                dst=os.path.join(config.DIRS['processed'][stage], tile),
            )
            shutil.copyfile(
                src=os.path.join(config.DIRS['processed']['tile_annotations'], tile),
                dst=os.path.join(config.DIRS['processed'][stage], f'target_{tile}'),
            )
    config.DIRS['processed']['tiles']

if __name__ == '__main__':
    # DeepLab v2 setup
    generate_train_val()
    for stage in ['train', 'val']:
        shutil.copytree(
            src=config.DIRS['processed'][stage],
            dst=os.path.join(config.XSEDE_V2_DIR, 'data', stage),
        )
    shutil.copyfile(
        src=config.MODEL_DEEPLABV2_PATH,
        dst=os.path.join(config.XSEDE_V2_DIR, 'model', 'deeplabv2_resnet101_msc-cocostuff164k-100000.pth'),
    )

    # DeepLab v3 setup
    shutil.copytree(
        src=config.DIRS['processed']['tiles'],
        dst=os.path.join(config.XSEDE_V3_DIR, 'data', 'tiles'),
    )
    shutil.copytree(
        src=config.DIRS['processed']['tile_annotations'],
        dst=os.path.join(config.XSEDE_V3_DIR, 'data', 'tile_annotations'),
    )

