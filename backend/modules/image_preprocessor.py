import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self):
        pass

    def __do_guassian_blur(self):
        pass

    def __find_right_side_up(self):
        pass

    @staticmethod
    def set_up_image(result):
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 2)
        _,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        if np.mean(thresh[:5, :]) > 200:
            thresh = cv2.bitwise_not(thresh)

        # h, w = thresh.shape
        thresh[:3, :] = 0
        thresh[-3:, :] = 0
        thresh[:, :3] = 0
        thresh[:, -3:] = 0

        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts,key=cv2.contourArea)

        card_contour = cnts[-1]
        rect = cv2.minAreaRect(card_contour)
        points = cv2.boxPoints(rect)

        # width, height = image.width, image.height
        width, height = result.shape[0], result.shape[1]
        dst_pts = np.float32([[0, 0],
                            [width-1, 0],
                            [width-1, height-1],
                            [0, height-1]])

        M = cv2.getPerspectiveTransform(np.float32(points), dst_pts)
        warped = cv2.warpPerspective(result, M, (width, height))

        # if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped,cv2.ROTATE_90_CLOCKWISE)
        return warped
