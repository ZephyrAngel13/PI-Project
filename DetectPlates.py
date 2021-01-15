# DetectPlates.py

import cv2
import numpy as np
import math
import Main
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar


PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


def detectPlatesInScene(imageOriginalScene):
    listOfPossiblePlates = []                   # va returna o lista cu posibilele placute

    height, width, numChannels = imageOriginalScene.shape

    imageGrayScene = np.zeros((height, width, 1), np.uint8)
    imageThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()                         #o folosim pentru a dealoca spatiile de memorie

    if Main.showSteps == True:
        cv2.imshow("Original Image-1", imageOriginalScene)


    imageGrayScene, imageThreshScene = Preprocess.preprocess(imageOriginalScene)         # procesul de a trece imaginea prin filtre

    if Main.showSteps == True:
        cv2.imshow("Gray Image-2", imageGrayScene)
        cv2.imshow("Thresh Image-3", imageThreshScene)


            # gasim toate caracterele posibile
            # gasim prima oara toate contururile, abia dupa vedem daca pot fi caractere(fara sa le comparam cu altele inca)
    listOfPossibleCharsInScene = findPossibleCharsInScene(imageThreshScene)

    if Main.showSteps == True: 


        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        cv2.imshow("Image with contours", imgContours)

            # avand lista cu posibilele caractere, gasim caracterele care se potrivesc
            #fiecare potrivire va fi considerata o posibila placuta
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Main.showSteps == True:

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("Image with contours-2", imgContours)
    # end if # show steps #########################################################################

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   # pentru fiecare potrivire vrem sa extragem posibila placuta
        possiblePlate = extractPlate(imageOriginalScene, listOfMatchingChars)

        if possiblePlate.imagePlate is not None:                          # daca placuta a fost gasita o adaugam la lista cu posibilele placute
            listOfPossiblePlates.append(possiblePlate)





    if Main.showSteps == True:
        print("\n")


        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.SCALAR_RED, 2)

            cv2.imshow("Image with Contours", imgContours)



            cv2.imshow("PossiblePlate", listOfPossiblePlates[i].imagePlate)
            cv2.waitKey(0)



        cv2.waitKey(0)


    return listOfPossiblePlates


###################################################################################################
def findPossibleCharsInScene(imageThresh):
    listOfPossibleChars = []                #returnam o lista cu posibilele caractere

    intCountOfPossibleChars = 0

    imageThreshCopy = imageThresh.copy()           #facem o copie a imaginii trecuta prin filtre

    contours, npaHierarchy = cv2.findContours(imageThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # cautam contururile

    height, width = imageThresh.shape                                 #luam inaltimea si lungimea formei
    imgContours = np.zeros((height, width, 3), np.uint8)
    for i in range(0, len(contours)):                       # pentru fiecare contur

        if Main.showSteps == True:
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)


        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(possibleChar):                   # verificam daca conturul e un posibil caracter
            intCountOfPossibleChars = intCountOfPossibleChars + 1           # incrementam count
            listOfPossibleChars.append(possibleChar)                        # adaugam la lista

    return listOfPossibleChars

    if Main.showSteps == True: # show steps #######################################################
        print("\nstep 2 - len(contours) = " + str(len(contours)))  # 2362 with MCLRNF1 image
        print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))  # 131 with MCLRNF1 image
        cv2.imshow("2a", imgContours)
    # end if # show steps #########################################################################




###################################################################################################
def extractPlate(imageOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()           # returnam o posibila placuta

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sortam caracterele de la stanga la dreapta dupa pozitia lui x

            # calculam centrul placutei
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

            # calculam inaltimea si lungimea placutei
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight              #calculam inaltimea totala a caracterelor


    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)       #calculam media inaltimilor

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

            # calculam unghiul placutei si acolo unde e nevoie il corectam(ca sa putem obtine o placuta rotita)
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            #punem centrul, inaltimea, lungimea si unghiul intr-o variabila
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

            # urmatoarele functii le folosim sa facem rottirea

            # obtienm matricea de rotatie pentru unghi
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imageOriginal.shape      # ia inaltimea si lungimea imaginii originale

    imgRotated = cv2.warpAffine(imageOriginal, rotationMatrix, (width, height))       # roteste toata imaginea

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter)) #folosim tuple ca sa stocam mai multe lucruri intr-o singura variabila

    possiblePlate.imagePlate = imgCropped         # copiaza imaginea decupata in variabila cu posibila placuta

    return possiblePlate

