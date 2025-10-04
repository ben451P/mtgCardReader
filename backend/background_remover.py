import torch
from torchvision import transforms
import numpy as np
import cv2
from background_remover_model import U2NET

class BackgroundRemover:
    def __init__(self, image, model_path='/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardPriceReader/static/assets/saved_models/face_detection_cv2/u2net.pth'):
        self.model_path = model_path

        self.image = image.convert('RGB')
        self.image_tensor = self.__get_image_tensor()

        self.result = None

    def __get_image_tensor(self):
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        return transform(self.image).unsqueeze(0)
    

    def __load_model(self):
        net = U2NET(3, 1)
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(self.model_path))
            net.cuda()
        else:
            net.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        net.eval()
        return net

    def __postprocess_mask(self,pred):

        pred = pred.squeeze().cpu().data.numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred = (pred * 255).astype(np.uint8)
        pred = cv2.resize(pred, self.image.size, interpolation=cv2.INTER_LINEAR)
        _, mask = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
        return mask

    def __extract_foreground(self, mask):
        original_np = np.array(self.image)
        if mask.shape != original_np.shape[:2]:
            mask = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_3ch = cv2.merge([mask, mask, mask])
        foreground = cv2.bitwise_and(original_np, mask_3ch)
        return foreground

    def __find_bounding_box(self,mask):
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found in the mask.")
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h

    def __crop_to_card(self,image, mask):
        x, y, w, h = self.__find_bounding_box(mask)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image

    def remove_bg_main(self):
        net = self.__load_model()

        if torch.cuda.is_available():
            self.image_tensor = self.image_tensor.cuda()
        with torch.no_grad():
            d1, _, _, _, _, _, _ = net(self.image_tensor)

        mask = self.__postprocess_mask(d1)
        foreground = self.__extract_foreground(mask)
        
        foreground_np = np.array(foreground)
        
        cropped_card = self.__crop_to_card(foreground_np, mask)
        final = cv2.cvtColor(cropped_card, cv2.COLOR_RGB2BGR)
        
        self.result = final

    def return_result(self):
        return self.result

    def write_to_path(self,output_path):
        cv2.imwrite(output_path, self.result)
        print(f"Cropped card image saved to {output_path}")
