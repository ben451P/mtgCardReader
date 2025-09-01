import time
import json
from PIL import Image

from backend.background_remover import BackgroundRemover
from backend.ocr_pipeline import OCRPipeline
from backend.scryfall_api import ScryfallAPI

def main_process():
    #nned to train a small model to detect whether or not a playing card is in frame
    path = '/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardPriceReader/static/assets/test.png'
    start_time = time.perf_counter()

    image = Image.open(path)
    segmenter = BackgroundRemover(image=image)
    segmenter.remove_bg_main()
    segmented = segmenter.return_result()

    print(time.perf_counter() - start_time)

    with open("ocr_api_key.json", "r") as file:
        key = json.loads(file.read())["key"]

    ocr_pipeline = OCRPipeline(segmented,key)
    ocr_pipeline.ocr_main()
    card_name = ocr_pipeline.extract_card_name()

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