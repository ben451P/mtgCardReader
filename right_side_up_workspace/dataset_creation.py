import os, random
import cv2
import numpy as np
import torch

dataset_path = "/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/right_side_up_workspace/dataset1.pt"

all_img_paths = []
for folder in ["classified","dataset","misclassified"]:
    for file in os.listdir(folder):
        all_img_paths.append(os.path.join(folder,file))

# all_img_paths = random.sample(all_img_paths,100)

dataset = []
for i,path in enumerate(all_img_paths):
    img = cv2.imread(path)
    dataset.append(img)
    print(f"{i+1}/{len(all_img_paths)} images read into memory!")

labels = []
for i in range(len(dataset)):
    choice = random.randint(0,3)
    for _ in range(choice):
        dataset[i] = cv2.rotate(dataset[i],cv2.ROTATE_90_CLOCKWISE)
    dataset[i] = torch.tensor(cv2.resize(dataset[i], (256,256)),dtype=torch.float32)
    labels.append(torch.tensor(choice,dtype=torch.long))
    print(f"{i+1}/{len(all_img_paths)} images rotated!")

images_tensor = torch.stack(dataset)
labels_tensor = torch.stack(labels)

torch.save({'images': images_tensor, 'labels': labels_tensor}, dataset_path)