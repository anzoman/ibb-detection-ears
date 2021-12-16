import cv2
import math
import numpy as np


class Preprocess:
    def histogram_equalization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img

    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def gamma_correction_rgb(self, img):
        # convert img to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # compute gamma = log(mid*255)/log(mean)
        mid = 0.5
        mean = np.mean(gray)
        gamma = math.log(mid * 255) / math.log(mean)

        # do gamma correction
        return np.power(img, gamma).clip(0, 255).astype(np.uint8)

    def gamma_correction_hsv(self, img):
        # convert img to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        # compute gamma = log(mid*255)/log(mean)
        mid = 0.5
        mean = np.mean(val)
        gamma = math.log(mid * 255) / math.log(mean)

        # do gamma correction on value channel
        val_gamma = np.power(val, gamma).clip(0, 255).astype(np.uint8)

        # combine new value channel with original hue and sat channels
        hsv_gamma = cv2.merge([hue, sat, val_gamma])
        return cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

    def edge_enhancement_canny(self, img):
        # convert img to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # threshold to binary
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # find contours - write black over all small contours
        letter = morph.copy()
        cntrs = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        for c in cntrs:
            area = cv2.contourArea(c)
            if area < 100:
                cv2.drawContours(letter, [c], 0, (0, 0, 0), -1)

        # do canny edge detection
        return cv2.Canny(letter, 200, 200)

    def bilateral_filter(self, img):
        return cv2.bilateralFilter(img, 9, 75, 75)
