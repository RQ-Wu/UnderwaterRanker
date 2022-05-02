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

def adjust_gamma(image, gamma=0.7):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   gc_test = cv.LUT(image.astype(np.uint8), table.astype(np.uint8))   
   gc_test = cv.cvtColor(gc_test,cv.COLOR_BGR2RGB) / 255
   return gc_test

def HE(image):
    ce_test = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ce_test = clahe.apply(ce_test)
    ce_test = cv.cvtColor(ce_test,cv.COLOR_GRAY2BGR)
    #cv.imshow('CLAHE', ce_test)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    ce_test = cv.cvtColor(ce_test,cv.COLOR_BGR2RGB) / 255 

    return ce_test