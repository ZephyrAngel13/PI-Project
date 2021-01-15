# DetectChars.py
import os

import cv2
import numpy as np
import math
import random

import Main
import Preprocess
import PossibleChar



kNearest = cv2.ml.KNearest_create()

        # constantele ca sa vedem daca e caracter
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

        # constantele pentru a compara 2 caractere
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

        #alte constante
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

###################################################################################################
def loadKNNDataAndTrainKNN():
    allContoursWithData = []                # declaram listele
    validContoursWithData = []

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # citim clasificarea pt caractere
    except:                                                                                 # daca nu putem deschide documentul
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return False


    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # citim informatiile din fisier
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return False


    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    kNearest.setDefaultK(1)                                                             #k=1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    return True


###################################################################################################
def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:          #daca lista e goala
        return listOfPossiblePlates


            #exista cel putin o placuta in lista

    for possiblePlate in listOfPossiblePlates:          # pentru fiecare placuta posibila

        possiblePlate.imageGray, possiblePlate.imageThresh = Preprocess.preprocess(possiblePlate.imagePlate)     #o trecem prin filtre

        if Main.showSteps == True:
            cv2.imshow("Possible PLate", possiblePlate.imagePlate)
            cv2.imshow("Possible PLate Gray", possiblePlate.imageGray)
            cv2.imshow("Possible Plate Thresh", possiblePlate.imageThresh)


                # marim imaginea pentru a ne fi mai usor sa detectam caracterele
        possiblePlate.imageThresh = cv2.resize(possiblePlate.imageThresh, (0, 0), fx = 1.6, fy = 1.6)


        thresholdValue, possiblePlate.imageThresh = cv2.threshold(possiblePlate.imageThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Main.showSteps == True:
            cv2.imshow("Possible Plate Thresh New", possiblePlate.imageThresh)


                #cautam toate caracterele posibile in placuta

        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imageGray, possiblePlate.imageThresh)


                #cautam posibilele potriviri de caractere in placuta
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)


        if (len(listOfListsOfMatchingCharsInPlate) == 0):			#daca n-am gasit niciun grup de potrivir3


            possiblePlate.strChars = ""
            continue						#luam for de la inceput


        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                              #pentru fiecare lista de potriviri de caractere
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)        #sortam caracterele de la stanga la dreapta
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              #scoatem caracterele care nu se potrivesc


                #presupunem ca cea mai lunga lista de potriviri e placuta
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

                # cautam prin toti vectorii de potriviri pentru a ii sti indexul
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i



                # presupunem ca cea mai lunga lista e placuta
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imageThresh, longestListOfMatchingCharsInPlate)


    return listOfPossiblePlates


###################################################################################################
def findPossibleCharsInPlate(imageGray, imageThresh):
    listOfPossibleChars = []                        # ceea ce returnam
    contours = []
    imageThreshCopy = imageThresh.copy()

            # gaseste toate contururile
    contours, npaHierarchy = cv2.findContours(imageThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                        # for each contour
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):              # daca conturul poate fi caracter
            listOfPossibleChars.append(possibleChar)       # adaugam la lista posibilele caractere



    return listOfPossibleChars


###################################################################################################
def checkIfPossibleChar(possibleChar):
            # functia verifica daca un contur poate fi caracter
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


###################################################################################################
def findListOfListsOfMatchingChars(listOfPossibleChars):
    listOfListsOfMatchingChars = []                  # ce returnam

    for possibleChar in listOfPossibleChars:                        #pentru fiecare caracter posibil in lista de carcatere posibile
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        #gasim toate caracterele din lista care se potrivesc cu caracterul curent

        listOfMatchingChars.append(possibleChar)                # adaugam posibilul caracter la posibila lista de caractere

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     # daca posibila lista nu e suficient de lunga pentru a forma o placuta
            continue                            # luam bucla de la capat pentru a mai adauga caractere



                                                # lista contine grupuri de caractere
        listOfListsOfMatchingChars.append(listOfMatchingChars)      # adugam urmatorul grup possibil de caractere

        listOfPossibleCharsWithCurrentMatchesRemoved = []

                                                # remove the current list of matching chars from the big list so we don't use those same chars twice,
                                                # make sure to make a new big list for this since we don't want to change the original big list
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # recursive call

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # for each list of matching chars found by recursive call
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # add to our original list of lists of matching chars


        break



    return listOfListsOfMatchingChars


###################################################################################################
def findListOfMatchingChars(possibleChar, listOfChars):
            # functie pentru a gasit posbilie caractere si lista cu posibile carcatere
            # gasim grupurile de caractere(potrivirile)si le adaugam in lista mare
    listOfMatchingChars = []                #ce returnam(lista mare)

    for possibleMatchingChar in listOfChars:                # pentru fiecare caracter
        if possibleMatchingChar == possibleChar:    # verificam daca caracterul mai exista o data in lista
                                                    # nu il includem in lista deoarece inseamna ca ar trebui sa fie de 2 ori
            continue                                # nu adaungam in lista si luam de la capat bucla

                    # functii pentru a verifica daca caracterele se potrivesc
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

                # verificam daca caracterele se potrivesc
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        # daca caracterele se potrivesc adaugam caracterul la lista



    return listOfMatchingChars


###################################################################################################
# folosim teorema lui pitagora pentru a calcula dinstanta dintre 2 caractere
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))


###################################################################################################

def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                           # verificam sa nu se divida la zero daca centurl x e egal cu zero
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # calculam unghiul
    else:
        fltAngleInRad = 1.5708                          #daca e zero folosim direct acest numar


    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # calculam unghiul in grade

    return fltAngleInDeg


###################################################################################################

#prevenim sa introducem contururi inutile , cum ar fi la 0, ca poate lua interiorul sau. Ar trebui sa detectam doar cifra
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                #ceea ce returnam

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:        #daca cele 2 caractere nu sunt la fel
                                                                            #daca cele 2 caractere au aproape acelasi mijloc
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):

                                # cautam care caracter e mai mic ca sa il scoatem
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:              #daca caracterul nu a fost deja scos
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         #il scoatem

                    else:
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)


    return listOfMatchingCharsWithInnerCharRemoved


###################################################################################################
def recognizeCharsInPlate(imageThresh, listOfMatchingChars):
    strChars = ""               # valoarea pe care o returneaza

    height, width = imageThresh.shape

    imageThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sortam caracterele de la stanga la dreapta

    cv2.cvtColor(imageThresh, cv2.COLOR_GRAY2BGR, imageThreshColor)                     # facem imaginea thresh color pt a putea desena contururile

    for currentChar in listOfMatchingChars:                                         # pentru fiecare caracter in placuta
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imageThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)           # desenam un dreptunghi in jurul caracterului

                # scoatem caracterul 
        imgROI = imageThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           # redimensionam imaginea

        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        

        npaROIResized = np.float32(npaROIResized)               

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              # apelam functia

        strCurrentChar = str(chr(int(npaResults[0][0])))            # luam caracterul din rezultat

        strChars = strChars + strCurrentChar                        # adaugam caracterul la sir

    

    if Main.showSteps == True: 
        cv2.imshow("Image with rectangles", imageThreshColor)

    

    return strChars








