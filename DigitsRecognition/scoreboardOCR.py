import cv2
import os
import numpy as np
import random
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
from multiprocessing import Pool

def scoreImgOCR(image):
    scoreImg = image
    if scoreImg.size == 0:
        return ''
    ### Threshold Preprocessing
    scoreImg, = thresholdPreprocess([scoreImg])

    ### get Each Digit ROI
    ROIs = getDigitsROI(scoreImg)
    ### OCR
    return getPredictText(ROIs)

def scoreboardOCR(scoreboardImg, cropData):
    ### Get Score Position by Format
    scoreImgs = getScoreImgs(scoreboardImg, cropData)
    # predictTexts = []
    # for scoreImg in scoreImgs:
    #     predictTexts.append(scoreImgOCR(scoreImg))
    # return predictTexts
    with Pool(4) as p:
        return p.map(scoreImgOCR, scoreImgs)
    
def getScoreImgs(scoreboardImg, cropData):
    scoreImgs = []
    for idx, data in enumerate(cropData):
        # x, y, w, h = data
        # scoreImgs.append(scoreboardImg[y:y+h, x:x+w].copy())
        x1, y1, x2, y2 = data
        scoreImgs.append(scoreboardImg[y1:y2, x1:x2].copy())
    return scoreImgs

def thresholdPreprocess(images):
    for idx, image in enumerate(images):

        ### cvtColor to gray
        images[idx] = cv2.cvtColor(images[idx], cv2.COLOR_BGR2GRAY)

        ### threshold
        _, images[idx] = cv2.threshold(images[idx], 0, 255, cv2.THRESH_OTSU)

        ### convert to black background white word
        count_white = np.sum(images[idx] > 0)
        count_black = np.sum(images[idx] == 0)
        if count_white > count_black:
            images[idx] = 255 - images[idx]
    return images

def getDigitsROI(image):
    ImgH, ImgW = image.shape
    totalArea = ImgH*ImgW
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    ROIs = []
    xPos = []
    for idx, statistic in enumerate(stats):
        areaProportion = round(100*statistic[4]/totalArea, 2)
        if areaProportion > 5.0 and areaProportion < 50.0:
            x, y, w, h = statistic[0:4]
            ROI = image[y:y+h,x:x+w].copy()

            ### ROI resize
            parameter = 32/h
            ROI = cv2.resize(ROI, (int(w*parameter), int(h*parameter)), interpolation=cv2.INTER_NEAREST)

            ### ROI paddding
            yAxisPadding = max(0, int((52-h)/2))
            xAxisPadding = max(0, int((51-w)/2))
            ROI = cv2.copyMakeBorder(ROI, yAxisPadding, yAxisPadding, xAxisPadding, xAxisPadding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            ### convert to white background black word
            ROI = 255 - ROI

            ROIs.append(ROI)
            xPos.append(x)
    ### Sort ROIs
    sortedROIs = [ROI for _,ROI in sorted(zip(xPos,ROIs), key=lambda pair: pair[0])]
    return sortedROIs

def getPredictText(ROIs, lang="TableTennis", cong="--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789"):
    texts = []
    for ROI in ROIs:
        boxes = pytesseract.image_to_boxes(ROI, lang=lang, config=cong)
        for b in boxes.splitlines():
            b = b.split(' ')
            texts.append(b[0])
    if len(texts) == 0:
        return ''
    else:
        return ''.join(texts)

if __name__ == '__main__':
    # imagePath = r"../images/video1_classified/2_3_10_11/frame19750.png"
    # scoreboardImg = cv2.imread(imagePath)
    # print(scoreboardOCR(scoreboardImg, cropData=video1_cropData))
    
    scoreImgs = []
    imagePath = r"../images/video2_digits/baseline/originAvg/0"
    fileNames = os.listdir(imagePath)
    correct = 0
    total = 0
    for fileName in fileNames:
        image = cv2.imread(os.path.join(imagePath, fileName))
        image, = thresholdPreprocess([image])

        ROIs = getDigitsROI(image)
        if getPredictText(ROIs) == '0':
            correct = correct+1
        # else:
        #     cv2.imshow('ROIs[0]', ROIs[0])
        #     cv2.waitKey(0)
        total = total+1
    print('correct = ', correct)
    print('total = ', total)