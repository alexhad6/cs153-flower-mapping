import numpy as np
import torch
import cv2

# Local imports
import config
import utils
import data

# Get PyTorch device (CPU or GPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def run_model(model, raw_image):
    # Adapted from https://github.com/msminhas93/DeepLabv3FineTuning/blob/master/Analysis.ipynb
    img = raw_image.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    with torch.no_grad():
        a = model(torch.from_numpy(img).type(torch.FloatTensor)/255)
    label_map = a['out'].cpu().detach().numpy()[0][0] > 0.2
    return label_map

def generate_model_masks(model):
    total_true_positives = 0
    total_true_negatives = 0
    total_false_positives = 0
    total_false_negatives = 0
    sum_abs_error = 0
    sum_squared_error = 0
    total_flower_area = 0
    total_plant_area = 0
    for plant in data.plants:
        size = config.TILE_SIZE
        plant_image = utils.load_image(plant.masked_image_path)
        label_map = run_model(model, plant_image)
        mask = (label_map * 255).astype(np.uint8)
        utils.write_image(mask, plant.deeplabv3_mask_path)

        # Load plant data
        plant_data = utils.load_json(plant.data_path)
        flower_area = plant_data['flower_area']
        plant_area = plant_data['plant_area']

        # Update pixel counts
        h, w = mask.shape
        outside_pixels = (w * h) - plant_area
        annotation_mask = utils.load_image(plant.annotation_path)[:, :, 0]
        positives = np.count_nonzero(mask)
        true_positives = np.count_nonzero(cv2.bitwise_and(mask, annotation_mask))
        true_negatives = np.count_nonzero(cv2.bitwise_not(cv2.bitwise_or(mask, annotation_mask))) - outside_pixels
        total_true_positives += true_positives
        total_true_negatives += true_negatives
        total_false_positives += positives - true_positives
        total_false_negatives += plant_area - positives - true_negatives

        # Update flower area error sums
        model_flower_area = np.count_nonzero(mask)
        error = (model_flower_area - flower_area)
        sum_abs_error += abs(error)
        sum_squared_error += error ** 2
        total_flower_area += flower_area
        total_plant_area += plant_area

        # Load area back into plant data
        plant_data['model_flower_area'] = model_flower_area
        utils.write_json(plant_data, plant.data_path)

    # Mask statistics
    total = total_true_positives + total_true_negatives + total_false_positives + total_false_negatives
    accuracy = (total_true_positives + total_true_negatives) / total
    precision = total_true_positives / (total_true_positives + total_false_positives)
    recall = total_true_positives / (total_true_positives + total_false_negatives)
    F1 = 2 * (recall * precision) / (recall + precision)
    print('Flower mask')
    print(f"  {'Accuracy':<25} {accuracy:>10.2%}")
    print(f"  {'Precision':<25} {precision:>10.2%}")
    print(f"  {'Recall':<25} {recall:>10.2%}")
    print(f"  {'F1':<25} {F1:>10.2%}")
    print()

    # Area errors
    mean_flower_area = total_flower_area / len(data.plants)
    mean_plant_area = total_plant_area / len(data.plants)
    RMSE = (sum_squared_error / len(data.plants)) ** 0.5
    MAE = sum_abs_error / len(data.plants)
    print('Area')
    print(f"  {'mean flower area':<25} {mean_flower_area:>10.2f}")
    print(f"  {'mean plant area':<25} {mean_plant_area:>10.2f}")
    print(f"  {'RMSE':<25} {RMSE:>10.2f}")
    print(f"  {'MAE':<25} {MAE:>10.2f}")
    print(f"  {'RMSE / (mean flower area)':<25} {RMSE / mean_flower_area:>10.2%}")
    print(f"  {'MAE  / (mean flower area)':<25} {MAE / mean_flower_area:>10.2%}")
    print(f"  {'RMSE / (mean plant area)':<25} {RMSE / mean_plant_area:>10.2%}")
    print(f"  {'MAE  / (mean plant area)':<25} {MAE / mean_plant_area:>10.2%}")
    print()

if __name__ == '__main__':
    model = torch.load(config.DEEPLABV3_PATH, map_location=lambda storage, loc: storage)
    model.eval()
    generate_model_masks(model)
