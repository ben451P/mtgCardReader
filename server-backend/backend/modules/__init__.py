from .background_remove.background_remover import BackgroundRemover
from .image_preprocess.image_preprocessor import ImagePreprocessor
from .title_extract.title_extractor import TitleExtractor
from .scryfall_card_finder import ScryfallCardFinder

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

current_dir = Path.cwd()

def get_card_data_from_image(image: np.ndarray) -> bool:
    image_no_background = BackgroundRemover.main(image)

    bound = ImagePreprocessor.isolate_bounding_box(image_no_background)
    right_side_up = ImagePreprocessor.flip_right_side_up(bound)

    text = TitleExtractor.get_title(right_side_up)
    print(text)
    
    data = ScryfallCardFinder.find(text)

    return data