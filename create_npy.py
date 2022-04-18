import lmdb
import os
import skimage.io as io
import skimage.color as color
import numpy as np
import cv2
import base64

if not os.path.exists('../dataset/UIEB_all/Ucolor_npy'):
    os.mkdir('../dataset/UIEB_all/Ucolor_npy')
# for i, filename in enumerate(os.listdir('../dataset/UIEB_all/UIEB/raw-890')):
#     print(i+1, '/', 890, 'start handling ' + filename)
#     rgb_img = io.imread(os.path.join('../dataset/UIEB_all/UIEB/raw-890', filename))
#     lab_img = color.rgb2lab(rgb_img)
#     hsv_img = color.rgb2hsv(rgb_img)
#     gt_img = io.imread(os.path.join('../dataset/UIEB_all/UIEB/reference-890/reference-890', filename))
#     depth = io.imread(os.path.join('../dataset/UIEB_all/UIEB_depth/U890_gdcp', filename))[:, :, np.newaxis]

#     all_img = np.concatenate((rgb_img, lab_img, hsv_img, gt_img, depth), axis=2)
#     print(all_img.shape)
#     # print(os.path.join('../dataset/UIEB_all/Ucolor_npy', filename[:-4]))
#     np.save(os.path.join('../dataset/UIEB_all/Ucolor_npy', filename[:-4]), all_img)

#     print(i+1, '/', 890, filename + ' has been added into database successfully!')
#     print()

all_img = np.load(os.path.join('../dataset/UIEB_all/Ucolor_npy', '752_img_.npy'))
print(all_img.shape)