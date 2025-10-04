from io import BytesIO
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

class CardImage:
    def __init__(self, pil_image):
        self.image = pil_image.convert("RGB")
        self.image_tensor = self.__get_image_tensor()

    def __get_image_tensor(self):
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        return transform(self.image).unsqueeze(0)
    

    def check_size(self):
        img_byte_arr = BytesIO()
        self.image.save(img_byte_arr, format=self.image.format if self.image.format else 'png')
        return img_byte_arr.tell()