import os
import sys

import cv2

from detectors.cascade_detector.cascade_detector import CascadeDetector


class CascadeDetectorEars3(CascadeDetector):
    # This cascade detector detects ears.

    def __init__(self):
        super().__init__()

        self.ear_cascade = cv2.CascadeClassifier(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades/ears',
                         'haarcascade_both_ears_bevk.xml'))

        if self.ear_cascade.empty():
            raise IOError('Unable to load the left ear cascade classifier xml file')

    def detect(self, img):
        det_list = self.ear_cascade.detectMultiScale(img, self.scale_factor, self.min_neighbors)
        if len(det_list) == 0:
            return []
        return det_list


if __name__ == '__main__':
    fname = sys.argv[1]
    img = cv2.imread(fname)
    detector = CascadeDetectorEars3()
    detected_loc = detector.detect(img)
    for x, y, w, h in detected_loc:
        cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)
    cv2.imwrite(fname + '.detected.jpg', img)
