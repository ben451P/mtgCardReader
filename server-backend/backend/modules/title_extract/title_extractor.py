
from PIL import Image
import numpy as np
import cv2
import os

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TitleExtractor:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(base_dir,"model_weights")

    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    
    # optimize hyperparameters
    @staticmethod
    def __preprocess_image(image: np.ndarray) -> Image:
        image = Image.fromarray(image).convert("RGB")

        img = image.crop((45,15,3.78 * image.width/5,image.height/9))
        image_resized = np.array(img)

        image_bw = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        image_bw = cv2.GaussianBlur(image_bw, (5,5), 0)
        _, threshold_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_OTSU)

        img = Image.fromarray(threshold_img).convert("RGB")
        
        return img

    @staticmethod
    def get_title(image: np.ndarray) -> str:
        img = TitleExtractor.__preprocess_image(image)
        pixel_values = TitleExtractor.processor(img, return_tensors="pt").pixel_values

        generated_ids = TitleExtractor.model.generate(pixel_values)
        generated_text = TitleExtractor.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text