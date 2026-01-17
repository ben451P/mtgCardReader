from .background_remove.background_remover import BackgroundRemover

import os
import numpy as np
from pathlib import Path

current_dir = Path.cwd()

def get_scryfall_info_from_image(image: np.ndarray) -> dict:
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