import cv2
import numpy as np

urankerset_name_list = open('ranker_train_list.txt').readlines()
all = np.empty((2500, 150*10, 3))
now_h = 0
next_h = 0
for i in range(20):
    file_index = int(np.random.rand(1)*800)
    for j in range(10):
        img = cv2.imread('../../dataset/UIERank/' + urankerset_name_list[file_index * 10 + j].rstrip())
        h, w = img.shape[:2]
        img = cv2.resize(img, (150, int(150/w*h)))
        print(img.shape)
        next_h = int(150/w*h) + now_h
        all[now_h:next_h, j*150:(j+1)*150] = img
    now_h = next_h 
all = all[:next_h]
cv2.imwrite('all.png', all)