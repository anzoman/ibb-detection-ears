from detectors.detector import Detector


class CascadeDetector(Detector):
    def __init__(self):
        self.scale_factor = None
        self.min_neighbors = None

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

    def set_min_neighbours(self, min_neighbors):
        self.min_neighbors = min_neighbors

    def detect(self, img):
        pass

