# PossiblePlate.py

import cv2
import numpy as np


class PossiblePlate:

    # constructor
    def __init__(self):
        self.imagePlate = None
        self.imageGray = None                                 #folosim none ca sa declaram un obiect null
        self.imageThresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""







