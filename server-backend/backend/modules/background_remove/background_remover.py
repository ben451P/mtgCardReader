import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from .u2net import U2NET

class BackgroundRemover:
    def __load_model(model_path):
        net = U2NET(3, 1)
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_path))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_path, map_location='cpu'))
        net.eval()
        return net
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, "model_weights", "u2net.pth")
    net = __load_model(model_path=model_path)

    @staticmethod
    def main(image_np: np.ndarray) -> np.ndarray:
        image = Image.fromarray(image_np).convert('RGB')
        image_tensor = BackgroundRemover.__get_image_tensor(image)

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        with torch.no_grad():
            d1, _, _, _, _, _, _ = BackgroundRemover.net(image_tensor)

        mask = BackgroundRemover.__postprocess_mask(image, d1)
        foreground = BackgroundRemover.__extract_foreground(image, mask)
        
        # means that no background to segment
        try:
            cropped_card = BackgroundRemover.__crop_to_card(foreground, mask)
        except ValueError:
            print("No contours found in mask")
            return image_np
        
        final = cv2.cvtColor(cropped_card, cv2.COLOR_RGB2BGR)
        
        return final

    def __get_image_tensor(image: Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)

    def __postprocess_mask(image: Image, pred: torch.Tensor) -> np.ndarray:

        pred = pred.squeeze().cpu().data.numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred = (pred * 255).astype(np.uint8)
        pred = cv2.resize(pred, image.size, interpolation=cv2.INTER_LINEAR)
        _, mask = cv2.threshold(pred, 127, 255, cv2.THRESH_OTSU)
        return mask

    def __extract_foreground(image: Image, mask: np.ndarray) -> np.ndarray:
        original_np = np.array(image)
        if mask.shape != original_np.shape[:2]:
            mask = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_3ch = cv2.merge([mask, mask, mask])
        foreground = cv2.bitwise_and(original_np, mask_3ch)

        return foreground

    def __find_bounding_box(mask: np.ndarray):
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found in the mask.")
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h

    def __crop_to_card(image: np.ndarray, mask: np.ndarray):
        x, y, w, h = BackgroundRemover.__find_bounding_box(mask)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image