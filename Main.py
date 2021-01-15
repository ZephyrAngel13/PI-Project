# Main.py

import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate


SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = True
###################################################################################################
def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         #metoda de a detecta caracterele, cei mai apropiati k vecini

    if blnKNNTrainingSuccessful == False:
        print("\nerror: KNN traning was not successful\n")
        return
    # end if

    imageOriginalScene = cv2.imread("image1.jpg")               #deschide imaginea

    if imageOriginalScene is None:                            # daca nu se deschide imaginea
        print("\nerror: image not read from file \n\n")
        os.system("pause")
        return


    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imageOriginalScene)           #lista posibilelor placute

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    cv2.imshow("Image Original Scene", imageOriginalScene)            # arata imaginea originala

    if len(listOfPossiblePlates) == 0:                          # daca nu e gasita nicio placuta
        print("\nno license plates were detected\n")
    else:
                # ca sa intram pe aceasta ramura inseamna ca a fost gasita cel putin o posibila placuta

                #sortam lista descrescator(de la cele mai multe caractere la cele mai putine)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
                #luam posibila placuta cu cele mai multe caractere (prima de la sortarea facuta mai sus)

        licPlate = listOfPossiblePlates[0]

        cv2.imshow("Plate Image", licPlate.imagePlate)           # arata placuta
        cv2.imshow("Thresh Image", licPlate.imageThresh)         #placuta trecuta prin filtru thresh

        if len(licPlate.strChars) == 0:                     # daca nu a fost gasit niciun caracter in placuta
            print("\nno characters were detected\n\n")
            return
        cv2.waitKey(0)

        return

        drawGreenRectangleAroundPlate(imageOriginalScene, licPlate)             # deseneaza dreptunghiuri in jurul posibilelor placute



###################################################################################################
def drawGreenRectangleAroundPlate(imageOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # gaseste cele 4 puncte in jurul placutei pentru a desena dreptungiul

    cv2.line(imageOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_GREEN, 2)         # deseneaza 4 linii verzi
    cv2.line(imageOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_GREEN, 2)
    cv2.line(imageOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_GREEN, 2)
    cv2.line(imageOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_GREEN, 2)


###################################################################################################
if __name__ == "__main__":
    main()


















