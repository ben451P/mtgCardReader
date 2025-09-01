
import requests
import json
import os
from PIL import Image
import cv2
import numpy as np


class OCRPipeline:
    def __init__(self, image):
        self.path = "/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardPriceReader/output/temp/output.png"

        self.api_key = "K85739286388957"

        self.image = Image.fromarray(image)

    def check_size(self):
        size = os.stat(self.path).st_size
        if size > 1024*1000:
            return False
        return True

    def crop(self,im):
        width, height = im.size

        left, top, right, bottom = 0, 0, width, height/3
        im = im.crop((left, top, right, bottom))
        return im

    def ocr_main(self, source = "https://api.ocr.space/parse/image"):

        self.image = cv2.cvtColor(np.array(self.image), cv2.COLOR_BGR2GRAY)
        self.image = cv2.GaussianBlur(self.image, (5,5), 0)
        self.image = Image.fromarray(self.image)
        self.image.save(self.path, optimize=True)

        #reduce image
        while not self.check_size():
            self.image = self.crop(self.image)
            self.image.save(self.path, optimize=True)

        self.image = Image.open(self.path)
        
        #fetch ocr from api
        with open(self.path, 'rb') as image_file:
            response = requests.post(
                source,
                files={'filename': image_file},
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