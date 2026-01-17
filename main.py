import time
import json
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from backend.background_remover import BackgroundRemover
from backend.ocr_pipeline import OCRPipeline
from backend.scryfall_api import ScryfallAPI
from backend.image_preprocessor import ImagePreprocessor

def main_process():
    path = '/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/static/assets/test6.jpg'
    start_time = time.perf_counter()

    image = Image.open(path)
    segmenter = BackgroundRemover(image=image)
    segmenter.remove_bg_main()
    segmented = segmenter.return_result()

    p = ImagePreprocessor()
    print(type(segmented))
    segmented = p.set_up_image(segmented)
    print(type(segmented))

    plt.subplot(1, 2, 2)
    plt.title("Threshold Mask")
    plt.imshow(segmented)
    plt.axis('off')
    plt.show()


    print(time.perf_counter() - start_time)

    with open("ocr_api_key.json", "r") as file:
        key = json.loads(file.read())["key"]

    ocr_pipeline = OCRPipeline(segmented,key)
    ocr_pipeline.ocr_main()
    card_name = ocr_pipeline.extract_card_name()
    print(card_name)

    if not card_name:
        return False

    print(time.perf_counter() - start_time)

    scryfall_api = ScryfallAPI()
    scryfall_api.fetch(card_name)
    data = scryfall_api.return_result()

    print(time.perf_counter() - start_time)
    return data

if __name__ == "__main__":
    data = main_process()
    with open("/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/test.json", "w") as file:
        file.write(json.dumps(data.json(), indent=2))