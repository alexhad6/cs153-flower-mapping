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
    sum_abs_percent_error = 0
    sum_squared_percent_error = 0
    for plant in data.plants:
        image = utils.load_image(plant.masked_image_path)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(image_hsv, np.array(lower_thresh), np.array(upper_thresh))
        kernel = np.ones((3, 3), np.uint8)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
        utils.write_image(hsv_mask, plant.hsv_mask_path)

        # Update pixel counts
        h, w = hsv_mask.shape
        total_pixels = w * h
        true_mask = utils.load_image(plant.annotation_path)[:, :, 0]
        positives = np.count_nonzero(hsv_mask)
        true_positives = np.count_nonzero(cv2.bitwise_and(hsv_mask, true_mask))
        true_negatives = np.count_nonzero(cv2.bitwise_not(cv2.bitwise_or(hsv_mask, true_mask)))
        total_true_positives += true_positives
        total_true_negatives += true_negatives
        total_false_positives += positives - true_positives
        total_false_negatives += total_pixels - positives - true_negatives

        # Update sum of squared errors
        plant_data = utils.load_json(plant.data_path)
        hsv_flower_area = np.count_nonzero(hsv_mask)
        area_percent_error = (hsv_flower_area - plant_data['flower_area']) / plant_data['plant_area']
        sum_abs_percent_error += abs(area_percent_error)
        sum_squared_percent_error += area_percent_error ** 2

        # Load area back into plant data
        plant_data['hsv_flower_area'] = hsv_flower_area
        utils.write_json(plant_data, plant.data_path)

    total = total_true_positives + total_true_negatives + total_false_positives + total_false_negatives
    accuracy = (total_true_positives + total_true_negatives) / total
    precision = total_true_positives / (total_true_positives + total_false_positives)
    recall = total_true_positives / (total_true_positives + total_false_negatives)
    F1 = 2 * (recall * precision) / (recall + precision)
    print('Mask Error')
    print(f"  {'Accuracy':<10} {accuracy:>8.2%}")
    print(f"  {'Precision':<10} {precision:>8.2%}")
    print(f"  {'Recall':<10} {recall:>8.2%}")
    print(f"  {'F1':<10} {F1:>8.2%}")
    print()

    # RMSPE and MAPE
    RMSPE = (sum_squared_percent_error / len(data.plants)) ** 0.5
    MAPE = sum_abs_percent_error / len(data.plants)
    print('Area error')
    print(f"  {'RMSPE':<10} {RMSPE:>8.2%}")
    print(f"  {'MAPE':<10} {MAPE:>8.2%}")
    print()

if __name__ == '__main__':
    # show_hsv_histogram()
    generate_hsv_masks()
