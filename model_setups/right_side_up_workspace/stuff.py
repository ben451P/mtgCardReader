
import requests
import json
import os
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import time


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
        self.image = self.__crop(self.image)
        while not self.__check_size():
            self.image = self.__crop(self.image)

        buffer = BytesIO()
        self.image.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

        #fetch ocr from api
        files = {
        'filename': ('image.png', buffer, 'image/png')
        }
        def get_response():
            return requests.post(
                source,
                files=files,
                data={
                    'apikey': self.api_key,
                    'language': 'eng'
                }
            )
        check_response = lambda response: response.status_code == 200

        response = get_response()
        for i in range(2):
            if check_response(response):break
            print(f"API timeout, waiting it out, {i}/2 retries")
            print(response.json())
            time.sleep(10)
            response = get_response()
        
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
    
import requests
import json

class ScryfallAPI:
    def __init__(self):
        self.data = None

    def __repr__(self):
        return self.data

    def fetch(self, card_name):
        card_name = card_name.lower()
        card_name = card_name.split()
        card_name = "+".join(card_name)
        
        query = f"https://api.scryfall.com/cards/named?fuzzy={card_name}"
        data = requests.get(query)
        self.data = data

    def return_result(self):
        return self.data
        # return "Sorry no result yet"
    
    def write_to_json(self,file_name):
        if not self.data:return "No data yet!"
        with open(file_name, "w") as file:
            file.write(json.dumps(self.data.json(), indent=2))