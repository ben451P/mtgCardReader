
import requests
import json
import os
from io import BytesIO
from PIL import Image
import cv2
import numpy as np


class OCRPipeline:
    def __init__(self, image, api_key):
        self.api_key = api_key
        self.image = Image.fromarray(image)

    def __check_size(self):
        img_byte_arr = BytesIO()
        self.image.save(img_byte_arr, format=self.image.format if self.image.format else 'png')
        size = img_byte_arr.tell()
        if size > 1024*1000:
            return False
        return True

    def __crop(self,im):
        width, height = im.size

        left, top, right, bottom = 0, 0, width, height/3
        im = im.crop((left, top, right, bottom))
        return im

    def ocr_main(self, source = "https://api.ocr.space/parse/image"):

        self.image = cv2.cvtColor(np.array(self.image), cv2.COLOR_BGR2GRAY)
        self.image = cv2.GaussianBlur(self.image, (5,5), 0)
        self.image = Image.fromarray(self.image)

        #reduce image
        while not self.__check_size():
            self.image = self.__crop(self.image)

        buffer = BytesIO()
        self.image.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

        #fetch ocr from api
        files = {
        'filename': ('image.png', buffer, 'image/png')
        }

        response = requests.post(
            source,
            files=files,
            data={
                'apikey': self.api_key,
                'language': 'eng'
            }
        )

        result = response.json()
        self.data = result

    def write_to_json(self):
        with open("/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardPriceReader/ocr/ocr_test_api_request.json","w") as file:
            file.write(json.dumps(self.data, indent=4))

    def extract_card_name(self):
        # if not self.data: return "OCR Not Run!"
        text = self.data["ParsedResults"][0]["ParsedText"].split("\r\n")
        card_name = text[0]
        return card_name