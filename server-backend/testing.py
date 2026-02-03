from backend.modules import get_card_data_from_image

import cv2
import os

from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np
import json


register_heif_opener()

path = "/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/server-backend/tests/IMG_1465.HEIC"

image = cv2.imread(path)
image = np.array(Image.open(path))
got_image = get_card_data_from_image(image)

def test_system():
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    test_image_dir = os.path.join(BASE_DIR, "tests")
    image_names = os.listdir(test_image_dir)

    acc = 0
    for image_name in image_names:
        image_dir = os.path.join(test_image_dir, image_name)
        image = np.array(Image.open(image_dir))
        got_image = get_card_data_from_image(image)

        if got_image["obkect"] == "card": acc += 1
    
    acc = round(acc / len(image_names), 5)
    return acc

acc = test_system()
print(acc)