from PIL import Image
import numpy as np
import cv2
from datetime import datetime

from transformers import TrOCRProcessor,VisionEncoderDecoderModel

# Load from local directory
model_path = r"C:\Users\Ben\Desktop\VSCodeCoding\MTGPriceReader\model_files\finetuned7"  # Your saved folder

processor = TrOCRProcessor.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path)

path = r"C:\Users\Ben\Desktop\VSCodeCoding\MTGPriceReader\OCR_InHouse_Workspace\data116.png"

time = datetime.now()

### display code
image = Image.open(path).convert("RGB")

img = image.crop((45,45,3.78 * image.width/5,image.height/9))
image_resized = np.array(img)

image_bw = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
image_bw = cv2.GaussianBlur(image_bw, (5,5), 0)
_, threshold_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_OTSU)

img = Image.fromarray(threshold_img).convert("RGB")
###
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()

# Example inference
pixel_values = processor(img, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

print(datetime.now() - time)