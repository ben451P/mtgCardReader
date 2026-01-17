# to check the cropping for the text of the mtg card

import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

path = r"C:\Users\Ben\Desktop\VSCodeCoding\MTGPriceReader\OCR_InHouse_Workspace\data116.png"

image = Image.open(path).convert("RGB")

img = image.crop((45,45,3.78 * image.width/5,image.height/9))
image_resized = np.array(img)

image_bw = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
image_bw = cv2.GaussianBlur(image_bw, (3,3), 0)

image_bw = cv2.Canny(image_bw, threshold1=100,threshold2=200)

_, threshold_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_OTSU)
img = Image.fromarray(threshold_img).convert("RGB")
# cv2.polylines(image_resized, approx, isClosed=True, color=(255, 0, 0), thickness=20)
# cv2.fillPoly(image_resized, approx, color=(255, 0, 0))

plt.imshow(img)
# plt.imshow(img)
plt.show()