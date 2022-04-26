import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import config
import utils
import deeplab

# Adapted from https://github.com/kazuto1011/deeplab-pytorch/blob/master/demo.py

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_classes():
    classes = {}
    with open(config.COCOSTUFF_LABELS_PATH) as f:
        for line in f:
            label = line.rstrip().split('\t')
            classes[int(label[0])] = label[1].split(',')[0]
    return classes

def create_model():
    model = deeplab.DeepLabV2_ResNet101_MSC(n_classes=config.COCOSTUFF_N_CLASSES)
    state_dict = torch.load(config.MODEL_PATH, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def run_model(model, raw_image):
    image = raw_image.astype(np.float32)
    image -= np.array([
        config.COCOSTUFF_MEAN['b'],
        config.COCOSTUFF_MEAN['g'],
        config.COCOSTUFF_MEAN['r'],
    ])
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
    _, _, h, w = image_tensor.shape
    logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()
    postprocessor = deeplab.DenseCRF(
        iter_max=config.CRF['iter_max'],
        pos_xy_std=config.CRF['pos_xy_std'],
        pos_w=config.CRF['pos_w'],
        bi_xy_std=config.CRF['bi_xy_std'],
        bi_rgb_std=config.CRF['bi_rgb_std'],
        bi_w=config.CRF['bi_w'],
    )
    probs = postprocessor(raw_image, probs)
    label_map = np.argmax(probs, axis=0)
    return label_map

def plot_label_map(label_map, classes, raw_image):
    labels = np.unique(label_map)
    rows = np.floor(np.sqrt(len(labels) + 1))
    cols = np.ceil((len(labels) + 1) / rows)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(int(rows), int(cols), 1)
    ax.set_title('Input image')
    ax.imshow(raw_image[:, :, ::-1])
    ax.axis('off')
    for i, label in enumerate(labels):
        mask = label_map == label
        ax = plt.subplot(rows, cols, i + 2)
        ax.set_title(classes[label])
        ax.imshow(raw_image[..., ::-1])
        ax.imshow(mask.astype(np.float32), alpha=0.5)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    classes = get_classes()
    model = create_model()
    image = utils.load_image(config.TEST_IMAGE_PATHS['apple-tree'])
    label_map = run_model(model, image)
    plot_label_map(label_map, classes, image)
