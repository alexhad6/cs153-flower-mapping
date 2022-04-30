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
    fig, (h_ax, s_ax, v_ax) = plt.subplots(1, 3, sharey=True)
    h_ax.set_title('H', fontsize=20)
    s_ax.set_title('S', fontsize=20)
    v_ax.set_title('V', fontsize=20)
    width = 1
    alpha = 0.75
    xticks = [0, 75, 125, 200, 255]
    h_ax.bar(bins[:-1], h_hist_nonflower / h_hist_nonflower.max(), width=width, alpha=alpha)
    h_ax.bar(bins[:-1], h_hist / h_hist.max(), width=width, alpha=alpha)
    s_ax.bar(bins[:-1], s_hist_nonflower / s_hist_nonflower.max(), width=width, alpha=alpha)
    s_ax.bar(bins[:-1], s_hist / s_hist.max(), width=width, alpha=alpha)
    v_ax.bar(bins[:-1], v_hist_nonflower / v_hist_nonflower.max(), width=width, alpha=alpha)
    v_ax.bar(bins[:-1], v_hist / v_hist.max(), width=width, alpha=alpha)
    h_ax.set_xticks(xticks)
    s_ax.set_xticks(xticks)
    v_ax.set_xticks(xticks)
    h_ax.xaxis.set_tick_params(labelsize=13)
    h_ax.yaxis.set_tick_params(labelsize=13)
    s_ax.xaxis.set_tick_params(labelsize=13)
    v_ax.xaxis.set_tick_params(labelsize=13)
    plt.show()

if __name__ == '__main__':
    show_hsv_histogram()
