import cv2
import numpy as np
import torch.nn as nn
import torch, os

class CustomCNN(nn.Module):
    def __init__(self,hidden_nodes,output_nodes,conv_kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,conv_kernel_size,stride=1,padding=1)
        self.conv2 = nn.Conv2d(64,hidden_nodes,conv_kernel_size,stride=1,padding=1)
        self.fc1 = nn.Linear(hidden_nodes * 32 * 32,16)
        self.fc2 = nn.Linear(16,output_nodes)
        self.pooling = nn.MaxPool2d((2,2),stride=2)
        self.adaPool = nn.AdaptiveMaxPool2d((32,32))
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pooling(x)
        x = self.relu(self.conv2(x))
        x = self.pooling(x)
        x = self.adaPool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class ImagePreprocessor:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(base_dir,"model_weights",f"model_weights.pt")

    model = CustomCNN(128, 4)
    model.load_state_dict(torch.load(path, weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def flip_right_side_up(image: np.ndarray):
        image_np = cv2.resize(image, (256, 256))

        image_np = torch.tensor(image_np, dtype=torch.float32)

        image_np = image_np.unsqueeze(0)          # (1, 256, 256, 3)

        image_np = image_np.permute(0, 3, 1, 2)   # (1, 3, 256, 256)

        image_np = image_np.to(ImagePreprocessor.device)

        ImagePreprocessor.model.eval()
        with torch.no_grad():
            outputs = ImagePreprocessor.model(image_np)
            preds = torch.argmax(outputs, dim=1)

        for _ in range(preds.item()):
            image = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)

        return image

    # optimize hyperparameters
    @staticmethod
    def isolate_bounding_box(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 2)

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _, thresh = cv2.threshold(blurred, 0, 125, cv2.THRESH_BINARY)

        if np.mean(thresh[:5, :]) > 200:
            thresh = cv2.bitwise_not(thresh)

        thresh[:5, :] = 0
        thresh[-5:, :] = 0
        thresh[:, :5] = 0
        thresh[:, -5:] = 0

        # remove border
        kernel = np.ones((7,7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
        thresh = cv2.erode(thresh, np.ones((5,5), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        card_contour = max(cnts, key=cv2.contourArea)

        rect = cv2.minAreaRect(card_contour)
        points = cv2.boxPoints(rect)

        # shrink rectangle slightly (in case border remove didn't work)
        center = points.mean(axis=0)
        points = (points - center) * 0.97 + center

        h, w = image.shape[:2]
        dst_pts = np.float32([[1, 1], [w-1, 1], [w-1, h-1], [1, h-1]])

        M = cv2.getPerspectiveTransform(points.astype(np.float32), dst_pts)
        warped = cv2.warpPerspective(image, M, (w, h))
        
        return warped

