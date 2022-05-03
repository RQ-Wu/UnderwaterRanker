import numpy as np
import cv2 as cv
import os

def white_balance(img, percent=0.5):
#    img=cv.cvtColor(img, cv.COLOR_RGB2BGR)
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv.split(img):
        cumhist = np.cumsum(cv.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv.LUT(channel, lut.astype('uint8')))
    wb_test=cv.merge(out_channels)
    #cv.imshow('WB1', img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
#    img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return cv.cvtColor(wb_test, cv.COLOR_BGR2RGB) / 255

def white_balance_waternet(img, percent=0.5):
    b, g, r = cv.split(img)
    sum_b, sum_g, sum_r = b.sum(), g.sum(), r.sum()
    _max = max(sum_b, sum_g, sum_r)
    ratio = np.array([_max / sum_b, _max / sum_g, _max / sum_r])
    h, w, c = img.shape

    img_flatten = img.reshape(h * w, c).T.astype('float32')

    satLevel1 = 0.005 * ratio
    satLevel2 = 0.005 * ratio

    minmax = lambda x: (x - x.min()) / (x.max() - x.min())

    for i in range(3):
        q = [satLevel1[i], 1 - satLevel2[i]]
        tiles = np.quantile(img_flatten[i], q)
        img_flatten[i, :] = minmax(np.clip(img_flatten[i, :], tiles[0], tiles[1]))

    img = img_flatten.reshape(c, h, w).transpose(1,2,0)
    img = img[:, :, [2, 1, 0]]
    return img

def adjust_gamma(image, gamma=0.7):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   gc_test = cv.LUT(image.astype(np.uint8), table.astype(np.uint8))   
   gc_test = cv.cvtColor(gc_test,cv.COLOR_BGR2RGB) / 255.0
   return gc_test

def adjust_gamma_waternet(image, gamma=0.7):
   return (cv.cvtColor(image,cv.COLOR_BGR2RGB) / 255.0) ** gamma

def HE(image):
    ce_test = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ce_test = clahe.apply(ce_test)
    ce_test = cv.cvtColor(ce_test,cv.COLOR_GRAY2BGR)
    # cv.imshow('CLAHE', ce_test)
    # cv.waitKey(0)
    #cv.destroyAllWindows()
    ce_test = cv.cvtColor(ce_test,cv.COLOR_BGR2RGB) / 255.0 

    return ce_test

def HE_waternet(image):
    ce_test = cv.cvtColor(image,cv.COLOR_BGR2Lab)
    clahe = cv.createCLAHE(clipLimit=0.01 * 255, tileGridSize=(8,8))
    ce_test[:, :, 0] = clahe.apply(ce_test[:, :, 0])
    ce_test = cv.cvtColor(ce_test,cv.COLOR_Lab2BGR)
    # cv.imshow('CLAHE waternet', ce_test)
    # cv.waitKey(0)
    #cv.destroyAllWindows()
    ce_test = cv.cvtColor(ce_test,cv.COLOR_BGR2RGB) / 255.0 

    return ce_test

def preprocess(x):
    return {
        'wb': white_balance_waternet(x),
        'ce': HE_waternet(x),
        'gc': adjust_gamma_waternet(x),
        'x': x
    }

if __name__ == '__main__':
    if not os.path.exists('../dataset/UIEB_all/WaterNet_npy'):
        os.mkdir('../dataset/UIEB_all/WaterNet_npy')
    for i, filename in enumerate(os.listdir('../dataset/UIEB_all/UIEB/raw-890')):
        print(i+1, '/', 890, 'start handling ' + filename)
        img_test = cv.imread(os.path.join('../dataset/UIEB_all/UIEB/raw-890', filename), cv.IMREAD_COLOR)
        wb_test = white_balance(img_test)
        gc_test = adjust_gamma(img_test)
        ce_test = HE(img_test)
        rgb_img = cv.cvtColor(img_test, cv.COLOR_BGR2RGB) / 255.0
        gt_img = cv.imread(os.path.join('../dataset/UIEB_all/UIEB/reference-890/reference-890', filename), cv.IMREAD_COLOR)
        gt_img = cv.cvtColor(gt_img, cv.COLOR_BGR2RGB) / 255.0

        all_img = np.concatenate((rgb_img, wb_test, gc_test, ce_test, gt_img), axis=2)
        print(all_img.shape)
        # print(os.path.join('../dataset/UIEB_all/Ucolor_npy', filename[:-4]))
        np.save(os.path.join('../dataset/UIEB_all/WaterNet_npy', filename[:-4]), all_img.astype(np.float32))

        print(i+1, '/', 890, filename + ' has been added into database successfully!')
        print()

    all_img = np.load(os.path.join('../dataset/UIEB_all/Ucolor_npy', '752_img_.npy'))
    print(all_img.shape)

