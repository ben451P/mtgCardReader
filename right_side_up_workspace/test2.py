import matplotlib.pyplot as plt
import cv2
from backend.image_rotator import ImageRotator
img = cv2.imread("/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/right_side_up_workspace/dataset/data0.png")
for _ in range(0):
    img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
test = ImageRotator(img)
thing = test.return_oriented_image()
fig,ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(thing)
plt.show()