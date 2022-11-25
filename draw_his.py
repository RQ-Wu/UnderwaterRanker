from utils import build_historgram
import skimage.io as io
import torch
import matplotlib.pyplot as plt

image = io.imread('../dataset/UIERank/15704two_step_0.png') / 255.0
# image = io.imread('../dataset/UIERank/15704two_step_0.png') / 255.
# plt.figure(figsize=(16, 4))
colors=[(1, 0.1, 0.1), (0, 1, 0.2), (0, 0.5, 1)]
for i in range(3):
    plt.hist(image[:, :, i].flatten(), density=True, bins=64, facecolor=colors[i], edgecolor=colors[i])
    # plt.xticks([])
    # plt.yticks([])
    plt.axis('off')
    plt.savefig('histogram_' + str(i) + '.png')
    plt.cla()

