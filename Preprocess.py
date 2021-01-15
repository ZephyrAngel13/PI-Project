# Preprocess.py

import cv2
import numpy as np
import math


GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)                #folosim pentru a specifica deviatia standard pe x si y in scopul de a scoate zgomotele
ADAPTIVE_THRESH_BLOCK_SIZE = 19                     #dimensiunea suprafetei vecinului
ADAPTIVE_THRESH_WEIGHT = 9                          #media ponderata a vecinilor locali minus o anumita valoare

###################################################################################################
def preprocess(imageOriginal):
    imageGray = extractValue(imageOriginal)

    imageMaxContrastGray = maximizeContrast(imageGray)             #maximizam contrastul

    height, width = imageGray.shape                                   #luam inaltimea si lungimea formei

    imageBlur = np.zeros((height, width, 1), np.uint8)                #np.uint8=float64; returneaza o noua matrice de zero in functie inaltimea si lungimea formei

    imageBlur = cv2.GaussianBlur(imageMaxContrastGray, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)        #folosim metoda gaussiana pentru a scoate zgomotele din imagine

    imageThresh = cv2.adaptiveThreshold(imageBlur, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imageGray, imageThresh


###################################################################################################
def extractValue(imageOriginal):
    height, width, numChannels = imageOriginal.shape

    imageHSV = np.zeros((height, width, 3), np.uint8)

    imageHSV = cv2.cvtColor(imageOriginal, cv2.COLOR_BGR2HSV)         #cautam valoarea unei culori

    imgHue, imgSaturation, imgValue = cv2.split(imageHSV)

    return imgValue


###################################################################################################
def maximizeContrast(imageGray):                        #folosim metodele black hat si top hat
                                                        #balck-hat evidentiaza obiectele inchise pe un fundal luminos
                                                        #top-hat evidentiaza obiectele luminoase de interes pe un fundal inchis
    height, width = imageGray.shape

    imageTopHat = np.zeros((height, width, 1), np.uint8)
    imageBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))   #returneaza o matrice pentru dreptunghi

    imageTopHat = cv2.morphologyEx(imageGray, cv2.MORPH_TOPHAT, structuringElement)
    imageBlackHat = cv2.morphologyEx(imageGray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imageGray, imageTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imageBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat











