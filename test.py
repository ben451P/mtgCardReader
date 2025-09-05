from backend.background_remover import BackgroundRemover
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

time = datetime.now()
image = Image.open("/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/static/assets/test6.jpg")
remover = BackgroundRemover(image)
remover.remove_bg_main()
result = remover.return_result()

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 2)
_,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contour_vis = result.copy()

if np.mean(thresh[:5, :]) > 200:
    thresh = cv2.bitwise_not(thresh)

h, w = thresh.shape
thresh[:3, :] = 0
thresh[-3:, :] = 0
thresh[:, :3] = 0
thresh[:, -3:] = 0

cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts,key=cv2.contourArea)

card_contour = cnts[-1]
rect = cv2.minAreaRect(card_contour)
points = cv2.boxPoints(rect)

width, height = image.width, image.height
dst_pts = np.float32([[0, 0],
                      [width-1, 0],
                      [width-1, height-1],
                      [0, height-1]])

M = cv2.getPerspectiveTransform(np.float32(points), dst_pts)
warped = cv2.warpPerspective(result, M, (width, height))

if warped.shape[1] > warped.shape[0]:
    warped = cv2.rotate(warped,cv2.ROTATE_90_CLOCKWISE)

print(datetime.now() - time)


cv2.drawContours(contour_vis, cnts, -1, (0, 255, 0), 5)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Contours Overlay")
plt.imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Threshold Mask")
plt.imshow(warped)
plt.axis('off')
plt.show()
