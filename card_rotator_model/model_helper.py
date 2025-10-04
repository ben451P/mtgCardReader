import torch, os
import cv2
from .custom_cnn import CustomCNN

class ModelHelper:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.realpath(__file__))
        PATH = os.path.join(BASE_DIR,"model",f"model.pt")

        self.model = CustomCNN(128, 4)
        self.model.load_state_dict(torch.load(PATH, weights_only=True))

    def normalize_data(self,img):
        img = torch.tensor(cv2.resize(img, (256,256)),dtype=torch.float32)
        img = torch.unsqueeze(img,dim=0)
        return img

    def get_rotation(self,in_image):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        with torch.no_grad():
            in_image = in_image.to(device)
            outputs = self.model(in_image.permute(0, 3, 1, 2))
            _, pred = torch.max(outputs, dim=1)
        return pred[0]