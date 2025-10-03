from card_rotator_model import ModelHelper
import cv2
import numpy as np

class ImageRotator:
    def __init__(self, img):
        self.img = img
        self.helper = ModelHelper()

    def return_oriented_image(self):
        image = self.helper.normalize_data(self.img)
        rotations = self.helper.get_rotation(image)
        return_image = self.img
        for _ in range(rotations):
            return_image = cv2.rotate(return_image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        return return_image
        
