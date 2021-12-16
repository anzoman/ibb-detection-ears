import os
import sys

import cv2
import numpy as np

from detectors.cascade_detector.cascade_detector import CascadeDetector


class CascadeDetectorEars2(CascadeDetector):
    # This cascade detector detects ears.

    def __init__(self):
        super().__init__()

        self.left_ear_cascade = cv2.CascadeClassifier(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades/ears',
                         'haarcascade_leftear_cvitkovic.xml'))

        self.right_ear_cascade = cv2.CascadeClassifier(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades/ears',
                         'haarcascade_rightear_cvitkovic.xml'))

        if self.left_ear_cascade.empty():
            raise IOError('Unable to load the left ear cascade classifier xml file')

        if self.right_ear_cascade.empty():
            raise IOError('Unable to load the right ear cascade classifier xml file')

    def _detect_left_ear(self, img):
        return self.left_ear_cascade.detectMultiScale(img, self.scale_factor, self.min_neighbors)

    def _detect_right_ear(self, img):
        return self.right_ear_cascade.detectMultiScale(img, self.scale_factor, self.min_neighbors)

    def detect(self, img):
        det_list_left_ear = self._detect_left_ear(img)
        det_list_right_ear = self._detect_right_ear(img)

        if len(det_list_left_ear) == 0 and len(det_list_right_ear) == 0:
            return []
        elif len(det_list_left_ear) >= 1 and len(det_list_right_ear) == 0:
            return det_list_left_ear
        elif len(det_list_left_ear) == 0 and len(det_list_right_ear) >= 1:
            return det_list_right_ear
        else:
            return np.concatenate((det_list_left_ear, det_list_right_ear))


if __name__ == '__main__':
    fname = sys.argv[1]
    img = cv2.imread(fname)
    detector = CascadeDetectorEars2()
    detected_loc = detector.detect(img)
    for x, y, w, h in detected_loc:
        cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)
    cv2.imwrite(fname + '.detected.jpg', img)
