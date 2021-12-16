import os

import cv2
import torch

from detectors.detector import Detector


class YOLOv5DetectorEars4(Detector):
    # This YOLOv5 detector detects ears.

    def __init__(self):
        super().__init__()

        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'training', 'model_batch_16_epoch_300_v5s/weights/best.pt')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
        self.model.to('cpu')

    def detect(self, img):
        results = self.model(img)
        cord = results.xywh[0][:, :-2].to('cpu').numpy()
        cord_int = []
        for (x, y, w, h) in cord:
            cord_int.append([int(x), int(y), int(w), int(h)])
        return cord_int


if __name__ == '__main__':
    fname = "../../data/ears/test/0011.png"
    img = cv2.imread(fname)
    detector = YOLOv5DetectorEars4()
    detected_loc = detector.detect(img)

    for (x, y, w, h) in detected_loc:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv2.imshow('Ear detector', img)
    cv2.waitKey()
