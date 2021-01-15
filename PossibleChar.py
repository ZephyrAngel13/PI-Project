# PossibleChar.py

import cv2
import numpy as np
import math


class PossibleChar:

    # constructor
    def __init__(self, _contour):
        self.contour = _contour                 #folosim self ca sa putem accesa atributele si metodele clasei

        self.boundingRect = cv2.boundingRect(self.contour)

        [intX, intY, intWidth, intHeight] = self.boundingRect

        self.intBoundingRectX = intX
        self.intBoundingRectY = intY                          #x,y sunt coordonatele din stanga sus
        self.intBoundingRectWidth = intWidth                  #intWidth,intHeight lungimea si inaltimea
        self.intBoundingRectHeight = intHeight

        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))

        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)

        #definim functiile ca sa putem face un rotated rectangle , in asa fel incat sa aproximam cat mai bine conturul











