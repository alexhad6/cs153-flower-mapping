import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

import utils
import data

def show_hsv_histogram():
    num_bins = 256
    bins = np.linspace(0, num_bins, num_bins + 1)
    h_hist = np.zeros(num_bins)
    s_hist = np.zeros(num_bins)
    v_hist = np.zeros(num_bins)
    for plant in data.plants:
        image = utils.load_image(plant.masked_image_path)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        flower_mask = utils.load_image(plant.annotation_path)[:, :, 0]
        flower_pixel_locations = flower_mask.nonzero()
        flower_pixels_hsv = image_hsv[flower_pixel_locations]
        h_hist += np.histogram(flower_pixels_hsv[:, 0], bins=bins)[0]
        s_hist += np.histogram(flower_pixels_hsv[:, 1], bins=bins)[0]
        v_hist += np.histogram(flower_pixels_hsv[:, 2], bins=bins)[0]
    fig, (h_ax, s_ax, v_ax) = plt.subplots(1, 3)
    h_ax.set_title('H')
    s_ax.set_title('S')
    v_ax.set_title('V')
    h_ax.bar(bins[:-1], h_hist)
    s_ax.bar(bins[:-1], s_hist)
    v_ax.bar(bins[:-1], v_hist)
    plt.show()

def generate_hsv_masks(lower_thresh=[0, 0, 200], upper_thresh=[125, 75, 255]):
    total_true_positives = 0
    total_true_negatives = 0
    total_false_positives = 0
    total_false_negatives = 0
    sum_abs_error = 0
    sum_squared_error = 0
    total_flower_area = 0
    total_plant_area = 0
    for plant in data.plants:
        image = utils.load_image(plant.masked_image_path)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(image_hsv, np.array(lower_thresh), np.array(upper_thresh))
        kernel = np.ones((3, 3), np.uint8)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
        utils.write_image(hsv_mask, plant.hsv_mask_path)

        # Load plant data
        plant_data = utils.load_json(plant.data_path)
        flower_area = plant_data['flower_area']
        plant_area = plant_data['plant_area']

        # Update pixel counts
        h, w = hsv_mask.shape
        outside_pixels = (w * h) - plant_area
        annotation_mask = utils.load_image(plant.annotation_path)[:, :, 0]
        positives = np.count_nonzero(hsv_mask)
        true_positives = np.count_nonzero(cv2.bitwise_and(hsv_mask, annotation_mask))
        true_negatives = np.count_nonzero(cv2.bitwise_not(cv2.bitwise_or(hsv_mask, annotation_mask))) - outside_pixels
        total_true_positives += true_positives
        total_true_negatives += true_negatives
        total_false_positives += positives - true_positives
        total_false_negatives += plant_area - positives - true_negatives

        # Update flower area error sums
        hsv_flower_area = np.count_nonzero(hsv_mask)
        error = (hsv_flower_area - flower_area)
        sum_abs_error += abs(error)
        sum_squared_error += error ** 2
        total_flower_area += flower_area
        total_plant_area += plant_area

        # Load area back into plant data
        plant_data['hsv_flower_area'] = hsv_flower_area
        utils.write_json(plant_data, plant.data_path)

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
    # show_hsv_histogram()
    generate_hsv_masks()
