import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
import base64
from PAI.service.persistence_service import save_preprocessed

def findCropInImage(img, template):
    template = getImgFromB64(template)
    img = getImgFromB64(img)

    try:
        w, h = template.shape[::-1]

        #methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                    #'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']


        method = eval('cv.TM_CCOEFF_NORMED')

        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img,top_left, bottom_right, 255, 2)
    except:
        return None
    
    
    return getB64FromImg(img)

def preprocess(joelhos):
    newImages = []

    for joelho in joelhos:
        img = getImgFromB64Internal(joelho.imagem)

        inverter = cv.flip(img, 1)

        newImages.append({
            "img": getB64FromImg(inverter),
            "type": joelho.tipo,
            "label": joelho.rotulo
        })

        img_equalizada = cv.equalizeHist(img)

        newImages.append({
            "img": getB64FromImg(img_equalizada),
            "type": joelho.tipo,
            "label": joelho.rotulo
        })


    save_preprocessed(newImages)

def getImgFromB64(b64):
    encoded_data = b64.split(',')

    if len(encoded_data) < 2:
        return None
    
    encoded_data = encoded_data[1]

    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv.imdecode(nparr, 0)

    return img

def getImgFromB64Internal(b64):
    encoded_data = b64

    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv.imdecode(nparr, 0)

    return img

def getB64FromImg(img):
    retval, buffer_img = cv.imencode('.png', img)
    data = base64.b64encode(buffer_img)
    img_str = data.decode("utf-8")

    return img_str