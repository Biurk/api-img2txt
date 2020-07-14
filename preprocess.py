from sklearn.preprocessing import StandardScaler
import sys
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFont, ImageOps
import re
from io import BytesIO
import base64
import requests
import json


# ce que fait ce script:
# 1: preprocess l'image
# 2: recup la prediction d'un model pour le nombre de ligne
# 3: segmente l'image en ligne grace a cette prediction
# 4: segmente chacune des lignes en mots grace a un model
# 5: recup la prediction d'un model pour chaque mot
# 6: renvoi le tout


# pour ne pas avoir d'elipse dans les print
np.set_printoptions(threshold=sys.maxsize)


def toGrayScale(img):
    return img.convert('L')


def isLightOnDark(pixels):
    return np.mean(pixels) < 128


def g(x):
    """ fonction sigmoid centré en 128, a pour effet d'ecraser vers 0 ce qui est en dessous de 128
    et vers 255 ce qui est au dessus"""
    return 1/((1/256)+np.exp(-0.04332169878499658*x))


def rescale(pixels, maxW, maxH):
    '''rescale si trop gros, pad si trop petit'''
    img = Image.fromarray(pixels)
    width = pixels.shape[1]
    height = pixels.shape[0]
    # on resize si l'image est trop grande on pad si trop petite
    if width > maxW or height > maxH:
        ratio = min(maxW/width, maxH/height)
        newsize = (round(width*ratio), round(height*ratio))
        # resize inverse height/width ????
        im = img.resize(newsize, Image.ANTIALIAS)
        pixels = np.array(im)
    pix = np.zeros((maxH, maxW))
    pix[:pixels.shape[0], :pixels.shape[1]] = pixels
    return pix


def preprocess(img):
    ''' retourne un pixel array directement exploitable en faisiant :
    1: transforme en grayscale
    2: transforme en text blanc sur noir
    4: augmente le contraste'''
    grayImg = toGrayScale(img)
    if not isLightOnDark(img):
        grayImg = ImageOps.invert(img)
    pixels = np.asarray(grayImg)
    return np.rint(g(g(pixels)))


# *********************** 1 *************************************
# PREPROCESS pour avoir un text en blanc sur noir tres contrasté
# **************************************************************

data = open('./img.txt', 'r').read()
image_data = re.sub('^data:image/.+;base64,', '', data)
img = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')
pixArray = preprocess(img)

# ************************ 2 ************************************
# Appel d'une premiere IA pour obtenir le nombre de lignes depuis le vecteur de projection horizontale de l'image
# **************************************************************


def get_projection(pixArray):
    sums = np.zeros(pixArray.shape[0])
    for x in range(len(pixArray)):
        sums[x] = sum(pixArray[x])
    return sums


scaledArray = rescale(pixArray, 1000, 500)
p = get_projection(scaledArray)
p = np.array(np.expand_dims(p, [0, 2, 3]), dtype='float32')

URL = "http://localhost:8401/v1/models/lineCounting:predict"
json_request = '{{ "instances" : {} }}'.format(np.array2string(
    p, separator=',', formatter={'float': lambda x: "%.1f" % x}))
r = requests.post(url=URL, data=json_request)
# extracting data in json format
data = r.json()
print('line number ', data)
line_pred = data['predictions'][0][0]

# ************************* 3 ***********************************
# On extrait la segmentation optimale a partir de l'indice de l'IA
# cette partie fait appelle a une transformée de fourier pour extraire la frequence dominante
# **************************************************************


def normalize(p):
    scaler = StandardScaler()
    sums = p.squeeze()
    sums = np.expand_dims(sums, 1)
    scaler.fit(sums)
    s = scaler.transform(sums)
    return np.squeeze(s)


def compute_global_offset(pixArray, freq):
    '''calcule le décalage des lignes optimal'''
    h = pixArray.shape[0]
    step = h/freq
    offset = 0
    best = float('inf')
    for guess in range(0, int(round(step/2))):
        current = 0
        y = guess
        while y < h:
            current += sum(pixArray[min(h-1, round(y))])
            y += step
        if current < best:
            best = current
            offset = guess
    return offset


def compute_segmentation(pixArray, freq, off=0):
    '''calcule la segmentation optimale'''
    h = pixArray.shape[0]
    step = h/freq
    result = []
    y = off
    while y < h:
        current = 0
        best = float('inf')
        offset = 0
        for guess in range(-round(0.3*step), round(0.3*step)):
            if y+guess < 0 or y+guess > h-1:
                continue
            current = sum(pixArray[max(min(h-1, round(y+guess)), 0)])
            if current <= best:
                best = current
                offset = guess
        result.append(round(y+offset))
        y += step+offset
    return result


def compute_frequency(p, hint):
    ''' recup la frequence la plus dominantes proche de l'indice fournis par une IA, grace a une FFT'''
    y = p
    w = np.abs(np.fft.rfft(y))  # on applique la fft
    # on regarde proche de l'indice
    guesses = w[max(1, int(0.8*hint)):min(len(w-1), int(1+1.2*hint))]
    return np.where(w == (np.max(np.abs(guesses))))[0][0]


def segmentImg(pixArray, segs):
    res = []
    prev = 0
    for x in segs:
        res.append(pixArray[prev:round(min(pixArray.shape[0], x+1))])
        prev = x
    return res


def full_process(initArray, proj, hint):
    p = normalize(proj)
    print(p.shape)
    freq = compute_frequency(p, hint).item()
    print('freq', freq)
    offset = compute_global_offset(pixArray, freq)
    return segmentImg(pixArray, compute_segmentation(pixArray, freq, offset))


segmented_lines = full_process(pixArray, get_projection(pixArray), line_pred)


# ************************ 4 ************************************
# Segmentation des lignes en mots grace a un model
# **************************************************************

def extractWords(line, wordsegPred):
    ''' extract words pixarray from line pixarray and word segmentation prediction'''
    p = wordsegPred > 0.5
    prev = False
    i = 0
    start = 0
    l = []
    for pp in p:
        if not prev and pp:
            start = i
        elif prev and not pp:
            l.append((start, i))
        prev = pp
        i += 1
    res = []
    for seg in l:
        res.append(line[:, seg[0]:seg[1]])
    return res


# rescale 35x1000
scaled_lines = []
for line in segmented_lines:
    scaled_lines.append(np.expand_dims(rescale(line, 1000, 35), 2))

scaled_lines = np.asarray(scaled_lines, dtype='float32')
scaled_lines = np.swapaxes(scaled_lines, 1, 2)

print(scaled_lines.shape)

# api-endpoint
URL = "http://localhost:8301/v1/models/wordSeg:predict"
# defining a params dict for the parameters to be sent to the API
json_request = '{{ "instances" : {} }}'.format(np.array2string(
    scaled_lines, separator=',', formatter={'float': lambda x: "%.1f" % x}))
# np.expand_dims(scaled_lines[0], 0), separator=',', formatter={'float': lambda x: "%.1f" % x}))
# sending get request and saving the response as response object
r = requests.post(url=URL, data=json_request)
# extracting data in json format
data = r.json()
wordSeg_preds = data['predictions']

print(np.asarray(wordSeg_preds).shape)
text = ''
nLine = 0
for wordsegPred in wordSeg_preds:
    segmented_words = extractWords(scaled_lines[nLine], wordsegPred)

    nLine += 1
