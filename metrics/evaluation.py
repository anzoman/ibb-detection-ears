import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score


class Evaluation:
    def convert2mask(self, mt, shape):
        # Converts coordinates of bounding-boxes into blank matrix with values set where bounding-boxes are.

        t = np.zeros([shape, shape])
        for m in mt:
            x, y, w, h = m
            cv2.rectangle(t, (x, y), (x + w, y + h), 1, -1)
        return t

    def prepare_for_detection(self, prediction, ground_truth):
        # For the detection task, convert Bounding-boxes to masked matrices (0 for background, 1 for the target).
        # If you run segmentation, do not run this function
        if len(prediction) == 0:
            return [], []

        # Large enough size for base mask matrices:
        shape = 2 * max(np.max(prediction), np.max(ground_truth))

        p = self.convert2mask(prediction, shape)
        gt = self.convert2mask(ground_truth, shape)

        return p, gt

    def iou_compute(self, p, gt):
        # Computes Intersection Over Union (IOU)
        if len(p) == 0:
            return 0

        intersection = np.logical_and(p, gt)
        union = np.logical_or(p, gt)

        iou = np.sum(intersection) / np.sum(union)

        return iou

    # Add your own metrics here, such as mAP, class-weighted accuracy, ...

    def tp(self, p, gt):
        return np.sum(np.logical_and(gt, p))

    def tn(self, p, gt):
        return np.sum(np.logical_not(np.logical_or(gt, p)))

    def fp(self, p, gt):
        return np.sum(np.logical_and(np.logical_not(gt), p))

    def fn(self, p, gt):
        return np.sum(np.logical_and(gt, np.logical_not(p)))

    def accuracy(self, tp, fp, fn, tn):
        total_gt = tp + fp + fn + tn
        if total_gt == 0:
            return 1.
        return float(tp + tn) / total_gt

    def precision(self, tp, fp, fn):
        total_predicted = tp + fp
        if total_predicted == 0:
            total_gt = tp + fn
            if total_gt == 0:
                return 1.
            else:
                return 0.
        return float(tp) / total_predicted

    def recall(self, tp, fn):
        total_gt = tp + fn
        if total_gt == 0:
            return 1.
        return float(tp) / total_gt

    def specificity(self, tn, fp):
        total_gt = tn + fp
        if total_gt == 0:
            return 1.
        return float(tn) / total_gt

    def f1_score(self, precision, recall):
        total_gt = precision + recall
        if total_gt == 0:
            return 1.
        return 2 * float(recall * precision) / total_gt

    def precision_recall_curve(self, y_true, iou_arr, thresholds):
        precisions = []
        recalls = []

        for threshold in thresholds:
            y_pred = ["positive" if score >= threshold else "negative" for score in iou_arr]

            precision = precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
            recall = recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")

            precisions.append(precision)
            recalls.append(recall)

        precisions.append(1)
        recalls.append(0)

        return precisions, recalls

    def ap(self, precisions, recalls):
        return np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
