# Adapted from https://github.com/kazuto1011/deeplab-pytorch/blob/master/demo.py

import os
import copy
import time
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import deeplab

COCOSTUFF_N_CLASSES = 182
COCOSTUFF_MEAN = {'r': 122.675, 'g': 116.669, 'b': 104.008}
TARGET_PREFIX = 'target'
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT_PATH, 'model', 'deeplabv2_resnet101_msc-cocostuff164k-100000.pth')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
CHECKPOINT_PATH = os.path.join(ROOT_PATH, 'deeplabv2.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FlowerDataset(datasets.VisionDataset):
    # Adapted from https://pytorch.org/vision/stable/_modules/torchvision/datasets/coco.html#CocoDetection

    def __init__(self, root):
        super().__init__(root)
        self.dir = os.path.join(DATA_PATH, self.root)
        self.images = [
            file_name
            for file_name in os.listdir(self.dir)
            if file_name.split('_')[0] != TARGET_PREFIX
        ]

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.dir, self.images[index]))
        image = image.astype(np.float32)
        image -= np.array([COCOSTUFF_MEAN['b'], COCOSTUFF_MEAN['g'], COCOSTUFF_MEAN['r']])
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().to(device)
        target = cv2.imread(os.path.join(self.dir, f'{TARGET_PREFIX}_{self.images[index]}'), cv2.IMREAD_GRAYSCALE)
        target = (target == 255).astype(np.uint8)
        target_tensor = torch.from_numpy(target).int().to(device)
        return image_tensor, target_tensor

    def __len__(self):
        return len(self.images)

def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.cpu().float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_F1 = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                # Freeze the batch norm pre-trained on COCO
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_true_positives = 0.0
            running_true_negatives = 0.0
            running_false_positives = 0.0
            running_false_negatives = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(inputs)
                    if phase == 'val':
                        # Wrap in list since validation phase only returns one
                        logits = [logits]
                    loss = 0
                    for logit in logits:
                        # Resize labels for {100%, 75%, 50%, Max} logits
                        _, _, h, w = logit.shape
                        labels_ = resize_labels(labels, size=(h, w))
                        loss += criterion(logit, labels_.to(device))

                    _, h, w = labels.shape
                    logits_full = F.interpolate(logits[-1], size=(h, w), mode="bilinear", align_corners=False)
                    probs = F.softmax(logits_full, dim=1)
                    preds = torch.argmax(probs, dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                all_positives = torch.sum(preds)
                true_positives = torch.sum((preds + labels) == 2)
                true_negatives = torch.sum((preds + labels) == 0)
                running_true_positives += true_positives
                running_true_negatives += true_negatives
                running_false_positives += all_positives - true_positives
                running_false_negatives += torch.numel(labels) - all_positives - true_negatives
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                epoch_loss /= 4  # Since there are four images
            total = running_true_positives + running_false_positives + running_true_negatives + running_false_negatives
            epoch_acc = (running_true_positives + running_true_negatives) / total
            epoch_precision = running_true_positives / (running_true_positives + running_false_positives)
            epoch_recall = running_true_positives / (running_true_positives + running_false_negatives)
            epoch_F1 = 2 * (epoch_recall * epoch_precision) / (epoch_recall + epoch_precision)
            print(f'{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  Precision: {epoch_precision:.4f}  Recall: {epoch_recall:.4f}  F1: {epoch_F1:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_F1 > best_F1:
                best_F1 = epoch_F1
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val F1: {best_F1:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # Set multiprocessing setting for GPU
    torch.multiprocessing.set_start_method('spawn')

    # Set up datasets
    stages = ['train', 'val']
    image_datasets = {stage: FlowerDataset(stage) for stage in stages}
    dataloaders = {
        stage: torch.utils.data.DataLoader(
            image_datasets[stage],
            batch_size=8,
            shuffle=True,
            num_workers=4,
        )
        for stage in stages
    }
    dataset_sizes = {stage: len(image_datasets[stage]) for stage in stages}
    print(dataset_sizes)

    # Set up model
    model = deeplab.DeepLabV2_ResNet101_MSC(n_classes=COCOSTUFF_N_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    model.base.aspp = deeplab.ASPP(n_classes=2)
    model.to(device)

    # Set up loss function
    class_weights = torch.FloatTensor([0.1, 1])
    class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion.to(device)

    # Set up optimizer
    optimizer = torch.optim.SGD(model.base.aspp.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train and save the model
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=30)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
