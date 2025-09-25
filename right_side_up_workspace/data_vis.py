import matplotlib.pyplot as plt
import os, random
import cv2
from .model import ScryfallAPI

base = "/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/right_side_up_workspace/dataset"
dataset = []
for path in os.listdir(base):
    for _ in range(10):
        img = cv2.imread(os.path.join(base,path))
        dataset.append(img)

labels = []
for i in range(len(dataset)):
    choice = random.randint(0,3)
    for _ in range(choice):
        dataset[i] = cv2.rotate(dataset[i],cv2.ROTATE_90_CLOCKWISE)
    dataset[i] = cv2.resize(dataset[i], (256,256))
    labels.append(choice)

fig, ax = plt.subplots(nrows=3,ncols=3)

ax = ax.flatten()

for i,img in enumerate(random.sample(dataset,9)):
    ax[i].imshow(img)

plt.show()

