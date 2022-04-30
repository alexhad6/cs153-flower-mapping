import numpy as np
import cv2
import matplotlib.pyplot as plt

# Local imports
import utils
import data

def show_hsv_histogram():
    num_bins = 256
    bins = np.linspace(0, num_bins, num_bins + 1)
    h_hist = np.zeros(num_bins)
    s_hist = np.zeros(num_bins)
    v_hist = np.zeros(num_bins)
    h_hist_nonflower = np.zeros(num_bins)
    s_hist_nonflower = np.zeros(num_bins)
    v_hist_nonflower = np.zeros(num_bins)
    for plant in data.plants:
        image = utils.load_image(plant.masked_image_path)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        plant_mask = utils.load_image(plant.mask_path)[:, :, 0]
        flower_mask = utils.load_image(plant.annotation_path)[:, :, 0]
        flower_pixel_locations = flower_mask.nonzero()
        flower_pixels_hsv = image_hsv[flower_pixel_locations]
        h_hist += np.histogram(flower_pixels_hsv[:, 0], bins=bins)[0]
        s_hist += np.histogram(flower_pixels_hsv[:, 1], bins=bins)[0]
        v_hist += np.histogram(flower_pixels_hsv[:, 2], bins=bins)[0]
        nonflower_pixel_locations = np.where(np.logical_and(flower_mask == 0, plant_mask != 0))
        nonflower_pixels_hsv = image_hsv[nonflower_pixel_locations]
        h_hist_nonflower += np.histogram(nonflower_pixels_hsv[:, 0], bins=bins)[0]
        s_hist_nonflower += np.histogram(nonflower_pixels_hsv[:, 1], bins=bins)[0]
        v_hist_nonflower += np.histogram(nonflower_pixels_hsv[:, 2], bins=bins)[0]
    fig, (h_ax, s_ax, v_ax) = plt.subplots(1, 3)
    h_ax.set_title('H')
    s_ax.set_title('S')
    v_ax.set_title('V')
    h_ax.bar(bins[:-1], h_hist_nonflower)
    h_ax.bar(bins[:-1], h_hist)
    s_ax.bar(bins[:-1], s_hist_nonflower)
    s_ax.bar(bins[:-1], s_hist)
    v_ax.bar(bins[:-1], v_hist_nonflower)
    v_ax.bar(bins[:-1], v_hist)
    plt.show()

if __name__ == '__main__':
    show_hsv_histogram()
